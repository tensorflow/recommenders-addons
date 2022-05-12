/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_GPU_H_
#define TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_GPU_H_

#include <typeindex>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/lookup_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/lib/nvhash/nv_hashtable.cuh"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/filebuffer.h"

namespace tensorflow {
namespace recommenders_addons {
namespace lookup {
namespace gpu {

using GPUDevice = Eigen::ThreadPoolDevice;

template <class V>
struct ValueArrayBase {};

template <class V, size_t DIM>
struct ValueArray : public ValueArrayBase<V> {
  V value[DIM];
};

template <class T>
using ValueType = ValueArrayBase<T>;

template <class K, class V>
class TableWrapperBase {
 public:
  virtual ~TableWrapperBase() {}
  virtual void upsert(const K* d_keys, const ValueType<V>* d_vals, size_t len,
                      cudaStream_t stream) {}
  virtual void accum(const K* d_keys, const ValueType<V>* d_vals_or_deltas,
                     const bool* d_exists, size_t len, cudaStream_t stream) {}
  virtual void dump(K* d_key, ValueType<V>* d_val, const size_t offset,
                    const size_t search_length, size_t* d_dump_counter,
                    cudaStream_t stream) const {}
  virtual void dump_to_file(OpKernelContext* ctx, const string filepath,
                            size_t dim, cudaStream_t stream,
                            const size_t buffer_size) const {}
  virtual void load_from_file(OpKernelContext* ctx, const string filepath,
                              const size_t key_num, size_t dim,
                              cudaStream_t stream,
                              const size_t buffer_size) const {}
  virtual void get(const K* d_keys, ValueType<V>* d_vals, bool* d_status,
                   size_t len, ValueType<V>* d_def_val, cudaStream_t stream,
                   bool is_full_size_default) const {}
  virtual size_t get_size(cudaStream_t stream) const {}
  virtual size_t get_capacity() const {}
  virtual void remove(const K* d_keys, size_t len, cudaStream_t stream) {}
  virtual void clear(cudaStream_t stream) {}
};

template <class K, class V, size_t DIM>
class TableWrapper final : public TableWrapperBase<K, V> {
 private:
  using Table = nv::HashTable<K, ValueArray<V, DIM>, ValueType<V>,
                              std::numeric_limits<K>::max(), DIM>;

 public:
  TableWrapper(size_t max_size) : max_size_(max_size) {
    table_ = new Table(max_size);
  }

  ~TableWrapper() override { delete table_; }

  void upsert(const K* d_keys, const ValueType<V>* d_vals, size_t len,
              cudaStream_t stream) override {
    table_->upsert(d_keys, d_vals, len, stream);
  }

  void accum(const K* d_keys, const ValueType<V>* d_vals_or_deltas,
             const bool* d_exists, size_t len, cudaStream_t stream) override {
    table_->accum(d_keys, d_vals_or_deltas, d_exists, len, stream);
  }

  void dump(K* d_key, ValueType<V>* d_val, const size_t offset,
            const size_t search_length, size_t* d_dump_counter,
            cudaStream_t stream) const override {
    table_->dump(d_key, d_val, offset, search_length, d_dump_counter, stream);
  }

  void dump_to_file(OpKernelContext* ctx, const string filepath, size_t dim,
                    cudaStream_t stream,
                    const size_t buffer_size) const override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    K* keys = nullptr;
    V* values = nullptr;
    size_t offset = 0;
    size_t* d_dump_counter;
    size_t dump_counter;
    size_t table_capacity = get_capacity();

    CUDA_CHECK(cudaMalloc(&keys, sizeof(K) * buffer_size));
    CUDA_CHECK(cudaMalloc(&values, sizeof(V) * buffer_size * dim));
    CUDA_CHECK(cudaMalloc(&d_dump_counter, sizeof(size_t)));

    string key_file = filepath + ".keys";
    string value_file = filepath + ".values";
    string key_tmpfile = filepath + ".keys.tmp";
    string value_tmpfile = filepath + ".values.tmp";
    auto key_buffer = filebuffer::DeviceFileBuffer<K>(key_tmpfile, buffer_size,
                                                      filebuffer::MODE::WRITE);
    auto value_buffer = filebuffer::DeviceFileBuffer<V>(
        value_tmpfile, buffer_size * dim, filebuffer::MODE::WRITE);
    size_t search_length = 0;

    size_t total_dumped = 0;
    while (offset < table_capacity) {
      if (offset + buffer_size >= table_capacity) {
        search_length = table_capacity - offset;
      } else {
        search_length = buffer_size;
      }
      table_->dump(keys, (ValueType<V>*)values, offset, search_length,
                   d_dump_counter, stream);
      CUDA_CHECK(cudaMemcpyAsync(&dump_counter, d_dump_counter, sizeof(size_t),
                                 cudaMemcpyDeviceToHost, stream));

      key_buffer.BatchPut(keys, dump_counter, stream);
      value_buffer.BatchPut(values, dump_counter * dim, stream);
      cudaStreamSynchronize(stream);
      offset += search_length;
      total_dumped += dump_counter;
    }

    LOG(INFO) << "Dump finish, offset=" << offset
              << ", total_dumped=" << total_dumped;

    CUDA_CHECK(cudaFree(keys));
    CUDA_CHECK(cudaFree(values));
    CUDA_CHECK(cudaFree(d_dump_counter));

    key_buffer.Close();
    value_buffer.Close();
    OP_REQUIRES(
        ctx, rename(key_tmpfile.c_str(), key_file.c_str()) == 0,
        errors::NotFound("key_tmpfile ", key_tmpfile, " is not found."));
    OP_REQUIRES(
        ctx, rename(value_tmpfile.c_str(), value_file.c_str()) == 0,
        errors::NotFound("value_tmpfile ", value_tmpfile, " is not found."));
  }

  void load_from_file(OpKernelContext* ctx, const string filepath,
                      const size_t key_num, size_t dim, cudaStream_t stream,
                      const size_t buffer_size) const override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    string key_file = filepath + ".keys";
    string value_file = filepath + ".values";
    auto key_buffer = filebuffer::DeviceFileBuffer<K>(key_file, buffer_size,
                                                      filebuffer::MODE::READ);
    auto value_buffer = filebuffer::DeviceFileBuffer<V>(
        value_file, buffer_size * dim, filebuffer::MODE::READ);

    size_t nkeys = 1;
    size_t total_keys = 0;
    size_t total_values = 0;
    while (nkeys > 0) {
      nkeys = key_buffer.Fill();
      value_buffer.Fill();
      total_keys += key_buffer.size();
      total_values += value_buffer.size();

      table_->upsert(key_buffer.data(), (ValueType<V>*)value_buffer.data(),
                     nkeys, stream);
      cudaStreamSynchronize(stream);
      key_buffer.Clear();
      value_buffer.Clear();
    }
    OP_REQUIRES(ctx, total_keys * dim == total_values,
                errors::DataLoss("load from file get invalid ", total_keys,
                                 " keys and", total_values, " values."));
  }

  void get(const K* d_keys, ValueType<V>* d_vals, bool* d_status, size_t len,
           ValueType<V>* d_def_val, cudaStream_t stream,
           bool is_full_size_default) const override {
    table_->get(d_keys, d_vals, d_status, len, d_def_val, stream,
                is_full_size_default);
  }

  size_t get_size(cudaStream_t stream) const override {
    return table_->get_size(stream);
  }

  size_t get_capacity() const override { return table_->get_capacity(); }

  void remove(const K* d_keys, size_t len, cudaStream_t stream) override {
    table_->remove(d_keys, len, stream);
  }

  void clear(cudaStream_t stream) override { table_->clear(stream); }

 private:
  size_t max_size_;
  Table* table_;
};

#define CREATE_A_TABLE(DIM)                                   \
  do {                                                        \
    if (runtime_dim == (DIM + 1)) {                           \
      *pptable = new TableWrapper<K, V, (DIM + 1)>(max_size); \
    };                                                        \
  } while (0)

#define CREATE_TABLE_PARTIAL_BRANCHES(PERIFX) \
  do {                                        \
    CREATE_A_TABLE((PERIFX)*10 + 0);          \
    CREATE_A_TABLE((PERIFX)*10 + 1);          \
    CREATE_A_TABLE((PERIFX)*10 + 2);          \
    CREATE_A_TABLE((PERIFX)*10 + 3);          \
    CREATE_A_TABLE((PERIFX)*10 + 4);          \
    CREATE_A_TABLE((PERIFX)*10 + 5);          \
    CREATE_A_TABLE((PERIFX)*10 + 6);          \
    CREATE_A_TABLE((PERIFX)*10 + 7);          \
    CREATE_A_TABLE((PERIFX)*10 + 8);          \
    CREATE_A_TABLE((PERIFX)*10 + 9);          \
  } while (0)

// create branches with dim range:
// [CENTILE * 100 + (DECTILE) * 10, CENTILE * 100 + (DECTILE) * 10 + 50]
#define CREATE_TABLE_BRANCHES(CENTILE, DECTILE)              \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 0); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 1); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 2); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 3); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 4);

template <class K, class V, int centile, int dectile>
void CreateTableImpl(TableWrapperBase<K, V>** pptable, size_t max_size,
                     size_t runtime_dim) {
  CREATE_TABLE_BRANCHES(centile, dectile);
}

#define DEFINE_CREATE_TABLE(ID, K, V, CENTILE, DECTILE)                      \
  void CreateTable##ID(size_t max_size, size_t runtime_dim,                  \
                       TableWrapperBase<K, V>** pptable) {                   \
    CreateTableImpl<K, V, CENTILE, DECTILE>(pptable, max_size, runtime_dim); \
  }

#define DECLARE_CREATE_TABLE(K, V)                       \
  void CreateTable0(size_t max_size, size_t runtime_dim, \
                    TableWrapperBase<K, V>**);           \
  void CreateTable1(size_t max_size, size_t runtime_dim, \
                    TableWrapperBase<K, V>**);           \
  void CreateTable2(size_t max_size, size_t runtime_dim, \
                    TableWrapperBase<K, V>**);           \
  void CreateTable3(size_t max_size, size_t runtime_dim, \
                    TableWrapperBase<K, V>**);

DECLARE_CREATE_TABLE(int64, float);
DECLARE_CREATE_TABLE(int64, Eigen::half);
DECLARE_CREATE_TABLE(int64, int64);
DECLARE_CREATE_TABLE(int64, int32);
DECLARE_CREATE_TABLE(int64, int8);

#undef CREATE_A_TABLE
#undef CREATE_DEFAULT_TABLE
#undef CREATE_TABLE_PARTIAL_BRANCHES
#undef CREATE_TABLE_ALL_BRANCHES
#undef DECLARE_CREATE_TABLE

}  // namespace gpu
}  // namespace lookup
}  // namespace recommenders_addons
}  // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_GPU_H_
