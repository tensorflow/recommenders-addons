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

#include <iomanip>
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
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/lib/nvhash/nv_hashtable.cuh"

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
  virtual size_t rehash_if_needed(const size_t min_capacity,
                                  cudaStream_t stream,
                                  const size_t new_keys_num = 0,
                                  const size_t last_hint_size = 0) {
    return 0;
  }
  virtual void get(const K* d_keys, ValueType<V>* d_vals, bool* d_status,
                   size_t len, ValueType<V>* d_def_val, cudaStream_t stream,
                   bool is_full_size_default) const {}
  virtual size_t get_size(cudaStream_t stream) const { return 0; }
  virtual size_t get_capacity() const { return 0; }
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

  size_t rehash_if_needed(const size_t min_capacity, cudaStream_t stream,
                          const size_t new_keys_num = 0,
                          const size_t last_hint_size = 0) override {
    K* d_keys;
    gpu::ValueArrayBase<V>* d_values;
    constexpr auto runtime_dim = DIM;
    size_t* d_dump_counter;
    constexpr const float max_load_factor = 0.75;
    constexpr const float min_load_factor = 0.25;

    size_t new_hint_size = last_hint_size + new_keys_num;
    const size_t capacity = table_->get_capacity();
    size_t new_capacity = capacity;
    const auto max_load_size = max_load_factor * capacity;
    const bool should_rehash = (new_hint_size > max_load_size);
    if (!should_rehash) {
      return new_hint_size;
    }

    const size_t total_size = table_->get_size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (total_size >= max_load_size) {
      new_capacity = capacity * 2;
    }
    if ((total_size < (min_load_factor * capacity)) &&
        (capacity > min_capacity)) {
      new_capacity = capacity / 2;
    }

    // The table should be able to hold new_keys_num at least
    if (new_capacity < new_hint_size) {
      new_capacity = new_hint_size * 2;
    }

    if (new_capacity != capacity) {  // rehash manually.
      size_t h_dump_counter = 0;
      CUDA_CHECK(cudaMallocManaged((void**)&d_dump_counter, sizeof(size_t)));
      CUDA_CHECK(cudaMallocManaged((void**)&d_keys, sizeof(K) * capacity));
      CUDA_CHECK(cudaMallocManaged((void**)&d_values,
                                   sizeof(V) * runtime_dim * capacity));
      table_->dump(d_keys, (gpu::ValueArrayBase<V>*)d_values, 0, capacity,
                   d_dump_counter, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      auto tmp_table_ = table_;
      table_ = new Table(new_capacity);
      delete tmp_table_;
      tmp_table_ = NULL;

      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaMemcpy((size_t*)&h_dump_counter, (size_t*)d_dump_counter,
                            sizeof(size_t), cudaMemcpyDefault));
      table_->upsert((const K*)d_keys, (const gpu::ValueArrayBase<V>*)d_values,
                     h_dump_counter, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaFree(d_keys));
      CUDA_CHECK(cudaFree(d_values));
      CUDA_CHECK(cudaFree(d_dump_counter));
      LOG(INFO) << "HashTable on GPU changes to new status: [size="
                << total_size << ", capacity=" << new_capacity
                << ", load factor=" << std::setprecision(2)
                << (float)total_size / (float)new_capacity << "].";
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    new_hint_size = table_->get_size(stream) + new_keys_num;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return new_hint_size;
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
};  // namespace gpu

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
DECLARE_CREATE_TABLE(int32, float);

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
