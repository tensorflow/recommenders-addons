/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#if GOOGLE_CUDA

#include "tensorflow_recommenders_addons/dynamic_embedding/core/kernels/cuckoo_hashtable_op_gpu.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/kernels/lookup_impl/lookup_table_op_gpu.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/utils.h"

#define EIGEN_USE_GPU

#include <cuda_runtime.h>
#include <stdlib.h>

#include <cstdlib>
#include <iomanip>
#include <type_traits>
#include <utility>

#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/stream_executor/stream.h"

namespace tensorflow {
namespace recommenders_addons {
namespace lookup {

using tensorflow::OpKernelContext;
using tensorflow::lookup::LookupInterface;

template <class K, class V>
class CuckooHashTableOfTensorsGpu final : public LookupInterface {
 public:
  CuckooHashTableOfTensorsGpu(OpKernelContext* ctx, OpKernel* kernel) {
    int64 init_size = 0;

    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "init_size", &init_size));

    if (init_size == 0) {
      int64 env_var = 0;
      Status status = ReadInt64FromEnvVar("TF_HASHTABLE_INIT_SIZE",
                                          1024 * 8,  // 8192 KV pairs by default
                                          &env_var);
      min_size_ = (size_t)env_var;
      max_size_ = (size_t)env_var;
    } else {
      min_size_ = init_size;
      max_size_ = init_size;
    }

    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(kernel->def(), "value_shape", &value_shape_));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(value_shape_),
        errors::InvalidArgument("Default value must be a vector, got shape ",
                                value_shape_.DebugString()));
    runtime_dim_ = value_shape_.dim_size(0);
    OP_REQUIRES(ctx, (runtime_dim_ <= 200),
                errors::InvalidArgument("The dim of HashTable on GPU should be "
                                        "less than or equal to 200, got ",
                                        runtime_dim_));
    this->CreateTable(max_size_, &table_);
    OP_REQUIRES(ctx, (table_ != nullptr),
                errors::InvalidArgument("HashTable on GPU is created failed!"));

    LOG(INFO) << "HashTable on GPU is created successfully:"
              << " K=" << std::type_index(typeid(K)).name()
              << ", V=" << std::type_index(typeid(V)).name()
              << ", max_size=" << max_size_ << ", min_size=" << min_size_;
  }

  ~CuckooHashTableOfTensorsGpu() { delete table_; }

  void CreateTable(size_t max_size, gpu::TableWrapperBase<K, V>** pptable) {
    if (runtime_dim_ <= 50) {
      gpu::CreateTable0(max_size, runtime_dim_, pptable);
    } else if (runtime_dim_ <= 100) {
      gpu::CreateTable1(max_size, runtime_dim_, pptable);
    } else if (runtime_dim_ <= 150) {
      gpu::CreateTable2(max_size, runtime_dim_, pptable);
    } else if (runtime_dim_ <= 200) {
      gpu::CreateTable3(max_size, runtime_dim_, pptable);
    } else {
      *pptable = nullptr;
    }
  }

  size_t size() const override {
    tf_shared_lock l(mu_);

    cudaStream_t _stream;
    CUDA_CHECK(cudaStreamCreate(&_stream));
    size_t retv = table_->get_size(_stream);
    CUDA_CHECK(cudaStreamSynchronize(_stream));
    CUDA_CHECK(cudaStreamDestroy(_stream));
    return retv;
  }

  Status Find(OpKernelContext* ctx, const Tensor& d_keys, Tensor* value,
              const Tensor& default_value) override {
    size_t len = d_keys.flat<K>().size();
    bool* d_status;
    gpu::ValueArrayBase<V>* d_default_value;

    auto value_flat = value->flat_inner_dims<V, 2>();
    const auto default_flat = default_value.flat<V>();
    int64 total = value_flat.size();
    int64 default_total = default_flat.size();
    bool is_full_default = (total == default_total);

    cudaStream_t _stream;

    if (len > 0) {
      size_t default_value_num =
          is_full_default ? default_value.shape().dim_size(0) : 1;
      CUDA_CHECK(cudaStreamCreate(&_stream));
      CUDA_CHECK(cudaMallocManaged((void**)&d_status, sizeof(bool) * len));
      {
        tf_shared_lock l(mu_);
        table_->get((const K*)d_keys.tensor_data().data(),
                    (gpu::ValueArrayBase<V>*)value->tensor_data().data(),
                    d_status, len,
                    (gpu::ValueArrayBase<V>*)default_value.tensor_data().data(),
                    _stream, is_full_default);
        CUDA_CHECK(cudaStreamSynchronize(_stream));
      }
      CUDA_CHECK(cudaFree(d_status));
      CUDA_CHECK(cudaFree(d_default_value));
      CUDA_CHECK(cudaStreamDestroy(_stream));
    }
    return Status::OK();
  }

  void RehashIfNeeded(cudaStream_t stream) {
    K* d_keys;
    gpu::ValueArrayBase<V>* d_values;
    size_t* d_dump_counter;
    size_t new_max_size = max_size_;

    size_t total_size = table_->get_size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (total_size >= 0.75 * max_size_) {
      new_max_size = max_size_ * 2;
    }
    if (total_size < 0.25 * max_size_ && max_size_ > min_size_) {
      new_max_size = max_size_ / 2;
    }
    if (new_max_size != max_size_) {  // rehash manually.
      size_t capacity = table_->get_capacity();
      size_t h_dump_counter = 0;
      CUDA_CHECK(cudaMallocManaged((void**)&d_dump_counter, sizeof(size_t)));
      CUDA_CHECK(cudaMallocManaged((void**)&d_keys, sizeof(K) * capacity));
      CUDA_CHECK(cudaMallocManaged((void**)&d_values,
                                   sizeof(V) * runtime_dim_ * capacity));
      table_->dump(d_keys, (gpu::ValueArrayBase<V>*)d_values, 0, capacity,
                   d_dump_counter, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      delete table_;
      table_ = NULL;
      CreateTable(new_max_size, &table_);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaMemcpy((size_t*)&h_dump_counter, (size_t*)d_dump_counter,
                            sizeof(size_t), cudaMemcpyDefault));
      table_->upsert((const K*)d_keys, (const gpu::ValueArrayBase<V>*)d_values,
                     h_dump_counter, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaFree(d_keys));
      CUDA_CHECK(cudaFree(d_values));
      CUDA_CHECK(cudaFree(d_dump_counter));
      max_size_ = new_max_size;
      LOG(INFO) << "HashTable on GPU changes to new status: [size="
                << total_size << ", max_size=" << max_size_
                << ", load factor=" << std::setprecision(2)
                << (float)total_size / (float)max_size_ << "].";
    }
  }

  Status Insert(OpKernelContext* ctx, const Tensor& keys,
                const Tensor& values) override {
    size_t len = keys.flat<K>().size();
    cudaStream_t _stream;
    CUDA_CHECK(cudaStreamCreate(&_stream));
    {
      mutex_lock l(mu_);
      RehashIfNeeded(_stream);
      table_->upsert((const K*)keys.tensor_data().data(),
                     (const gpu::ValueArrayBase<V>*)values.tensor_data().data(),
                     len, _stream);
      CUDA_CHECK(cudaStreamSynchronize(_stream));
    };
    CUDA_CHECK(cudaStreamDestroy(_stream));

    return Status::OK();
  }

  Status Remove(OpKernelContext* ctx, const Tensor& keys) override {
    size_t len = keys.flat<K>().size();
    K* d_keys;
    cudaStream_t _stream;

    CUDA_CHECK(cudaStreamCreate(&_stream));
    if (len > 0) {
      CUDA_CHECK(cudaMallocManaged((void**)&d_keys, sizeof(K) * len));
      CUDA_CHECK(cudaMemcpy((void*)d_keys, (void*)keys.tensor_data().data(),
                            sizeof(K) * len, cudaMemcpyDefault));
      {
        mutex_lock l(mu_);
        table_->remove((const K*)d_keys, len, _stream);
        RehashIfNeeded(_stream);
        CUDA_CHECK(cudaStreamSynchronize(_stream));
      }
      CUDA_CHECK(cudaStreamDestroy(_stream));
      CUDA_CHECK(cudaFree(d_keys));
    }
    return Status::OK();
  }

  Status Clear(OpKernelContext* ctx) {
    cudaStream_t _stream;
    CUDA_CHECK(cudaStreamCreate(&_stream));
    {
      mutex_lock l(mu_);
      table_->clear(_stream);
      RehashIfNeeded(_stream);
      CUDA_CHECK(cudaStreamSynchronize(_stream));
    }
    CUDA_CHECK(cudaStreamDestroy(_stream));
    return Status::OK();
  }

  Status ImportValues(OpKernelContext* ctx, const Tensor& keys,
                      const Tensor& values) override {
    size_t len = keys.flat<K>().size();
    K* d_keys;
    gpu::ValueArrayBase<V>* d_values;
    if (len > 0) {
      cudaStream_t _stream;
      CUDA_CHECK(cudaStreamCreate(&_stream));
      CUDA_CHECK(cudaMallocManaged((void**)&d_keys, sizeof(K) * len));
      CUDA_CHECK(
          cudaMallocManaged((void**)&d_values, sizeof(V) * runtime_dim_ * len));
      CUDA_CHECK(cudaMemcpy((void*)d_keys, (void*)keys.tensor_data().data(),
                            sizeof(K) * len, cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy((void*)d_values, (void*)values.tensor_data().data(),
                            sizeof(V) * runtime_dim_ * len, cudaMemcpyDefault));
      {
        mutex_lock l(mu_);
        table_->clear(_stream);
        table_->upsert((const K*)d_keys,
                       (const gpu::ValueArrayBase<V>*)d_values, len, _stream);
        CUDA_CHECK(cudaStreamSynchronize(_stream));
      }
      CUDA_CHECK(cudaStreamDestroy(_stream));
      CUDA_CHECK(cudaFree(d_keys));
      CUDA_CHECK(cudaFree(d_values));
    }
    return Status::OK();
  }

  Status ExportValues(OpKernelContext* ctx) override {
    size_t len = 0;
    int64 size = 0;

    const size_t offset = 0;

    Tensor* keys;
    Tensor* values;

    size_t* d_dump_counter;
    cudaStream_t _stream;
    CUDA_CHECK(cudaStreamCreate(&_stream));

    {
      tf_shared_lock l(mu_);
      len = table_->get_capacity();
      size = (int64)table_->get_size(_stream);
      CUDA_CHECK(cudaStreamSynchronize(_stream));
    }

    CUDA_CHECK(cudaMallocManaged((void**)&d_dump_counter, sizeof(size_t)));

    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    attr.set_on_host(false);

    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({(size)}), &keys, attr));
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "values", TensorShape({size, (int64)runtime_dim_}), &values, attr));
    if (size) {
      tf_shared_lock l(mu_);
      table_->dump((K*)keys->flat<K>().data(),
                   (gpu::ValueArrayBase<V>*)values->matrix<V>().data(), offset,
                   len, d_dump_counter, _stream);
      CUDA_CHECK(cudaStreamSynchronize(_stream));
    }
    CUDA_CHECK(cudaFree(d_dump_counter));
    CUDA_CHECK(cudaStreamDestroy(_stream));
    return Status::OK();
  }

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }
  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }
  TensorShape key_shape() const final { return TensorShape(); }
  TensorShape value_shape() const override { return value_shape_; }

 private:
  TensorShape value_shape_;
  size_t max_size_;
  size_t min_size_;
  size_t runtime_dim_;
  mutable mutex mu_;
  gpu::TableWrapperBase<K, V>* table_ = nullptr GUARDED_BY(mu_);
};

}  // namespace lookup

// Table lookup op. Perform the lookup operation on the given table.
class HashTableFindGpuOp : public OpKernel {
 public:
  explicit HashTableFindGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    // Input 0 could be a STRING_REF or a RESOURCE
    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor& key = ctx->input(1);
    const Tensor& default_value = ctx->input(2);

    TensorShape output_shape = key.shape();
    output_shape.RemoveLastDims(table->key_shape().dims());
    output_shape.AppendShape(table->value_shape());
    Tensor* out;
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("values", output_shape, &out, attr));

    OP_REQUIRES_OK(ctx, table->Find(ctx, key, out, default_value));
  }
};

REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(CuckooHashTableFind)).Device(DEVICE_GPU),
    HashTableFindGpuOp);

// Table insert op.
class HashTableInsertGpuOp : public OpKernel {
 public:
  explicit HashTableInsertGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForInsert(keys, values));
    OP_REQUIRES_OK(ctx, table->Insert(ctx, keys, values));
  }
};

REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(CuckooHashTableInsert)).Device(DEVICE_GPU),
    HashTableInsertGpuOp);

// Table remove op.
class HashTableRemoveGpuOp : public OpKernel {
 public:
  explicit HashTableRemoveGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& key = ctx->input(1);
    OP_REQUIRES_OK(ctx, table->CheckKeyTensorForRemove(key));
    OP_REQUIRES_OK(ctx, table->Remove(ctx, key));
  }
};

REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(CuckooHashTableRemove)).Device(DEVICE_GPU),
    HashTableRemoveGpuOp);

// Table clear op.
template <class K, class V>
class HashTableClearGpuOp : public OpKernel {
 public:
  explicit HashTableClearGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);
    lookup::CuckooHashTableOfTensorsGpu<K, V>* table_cuckoo =
        (lookup::CuckooHashTableOfTensorsGpu<K, V>*)table;
    OP_REQUIRES_OK(ctx, table_cuckoo->Clear(ctx));
  }
};

// Op that returns the size of the given table.
class HashTableSizeGpuOp : public OpKernel {
 public:
  explicit HashTableSizeGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    Tensor* out;
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_on_host(false);

    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("size", TensorShape({}), &out, attr));

    size_t size = table->size();
    const int64* p_size = (const int64*)out->flat<int64>().data();
    CUDA_CHECK(cudaMemcpy((void*)out->tensor_data().data(), (void*)&size,
                          sizeof(size_t), cudaMemcpyDefault));
  }
};

REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(CuckooHashTableSize)).Device(DEVICE_GPU),
    HashTableSizeGpuOp);

// Op that outputs tensors of all keys and all values.
class HashTableExportGpuOp : public OpKernel {
 public:
  explicit HashTableExportGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, table->ExportValues(ctx));
  }
};

REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(CuckooHashTableExport)).Device(DEVICE_GPU),
    HashTableExportGpuOp);

// Clear the table and insert data.
class HashTableImportGpuOp : public OpKernel {
 public:
  explicit HashTableImportGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 =
        (ctx->input_dtype(0) == DT_RESOURCE) ? DT_RESOURCE : DT_STRING_REF;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForImport(keys, values));
    OP_REQUIRES_OK(ctx, table->ImportValues(ctx, keys, values));
  }
};

REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(CuckooHashTableImport)).Device(DEVICE_GPU),
    HashTableImportGpuOp);

// Register the CuckooHashTableOfTensors op.

#define REGISTER_KERNEL(key_dtype, value_dtype)                            \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name(PREFIX_OP_NAME(CuckooHashTableOfTensors))                       \
          .Device(DEVICE_GPU)                                              \
          .TypeConstraint<key_dtype>("key_dtype")                          \
          .TypeConstraint<value_dtype>("value_dtype"),                     \
      HashTableGpuOp<                                                      \
          lookup::CuckooHashTableOfTensorsGpu<key_dtype, value_dtype>,     \
          key_dtype, value_dtype>);                                        \
  REGISTER_KERNEL_BUILDER(Name(PREFIX_OP_NAME(CuckooHashTableClear))       \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<key_dtype>("key_dtype")      \
                              .TypeConstraint<value_dtype>("value_dtype"), \
                          HashTableClearGpuOp<key_dtype, value_dtype>)

REGISTER_KERNEL(int64, float);
REGISTER_KERNEL(int64, Eigen::half);
REGISTER_KERNEL(int64, int32);
REGISTER_KERNEL(int64, int8);

#undef REGISTER_KERNEL

}  // namespace recommenders_addons
}  // namespace tensorflow
#endif
