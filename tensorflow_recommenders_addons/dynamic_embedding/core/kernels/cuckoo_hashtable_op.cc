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

#define EIGEN_USE_THREADS

#include "tensorflow_recommenders_addons/dynamic_embedding/core/kernels/cuckoo_hashtable_op.h"

#include <string>
#include <type_traits>
#include <utility>

#include "tensorflow/core/kernels/lookup_table_op.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/kernels/lookup_impl/lookup_table_op_cpu.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/types.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/utils.h"

namespace tensorflow {
namespace recommenders_addons {
namespace lookup {
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, class K, class V>
struct LaunchTensorsFind;

template <class K, class V>
struct LaunchTensorsFind<CPUDevice, K, V> {
  explicit LaunchTensorsFind(int64 value_dim) : value_dim_(value_dim) {}

  void launch(OpKernelContext* context, cpu::TableWrapperBase<K, V>* table,
              const Tensor& key, Tensor* value, const Tensor& default_value) {
    const auto key_flat = key.flat<K>();
    cpu::Tensor2D<V> value_flat = value->flat_inner_dims<V, 2>();
    cpu::ConstTensor2D<V> default_flat = default_value.flat_inner_dims<V, 2>();
    int64 total = value_flat.size();
    int64 default_total = default_flat.size();
    bool is_full_default = (total == default_total);

    auto shard = [this, table, key_flat, &value_flat, &default_flat,
                  &is_full_default](int64 begin, int64 end) {
      for (int64 i = begin; i < end; ++i) {
        if (i >= key_flat.size()) {
          break;
        }
        table->find(key_flat(i), value_flat, default_flat, value_dim_,
                    is_full_default, i);
      }
    };
    auto& worker_threads = *context->device()->tensorflow_cpu_worker_threads();
    int64 slices = static_cast<int64>(total / worker_threads.num_threads) + 1;
    Shard(worker_threads.num_threads, worker_threads.workers, total, slices,
          shard);
  }

 private:
  const int64 value_dim_;
};

template <typename Device, class K, class V>
struct LaunchTensorsInsert;

template <class K, class V>
struct LaunchTensorsInsert<CPUDevice, K, V> {
  explicit LaunchTensorsInsert(int64 value_dim) : value_dim_(value_dim) {}

  void launch(OpKernelContext* context, cpu::TableWrapperBase<K, V>* table,
              const Tensor& keys, const Tensor& values) {
    const auto key_flat = keys.flat<K>();
    int64 total = key_flat.size();
    const auto value_flat = values.flat_inner_dims<V, 2>();

    auto shard = [this, &table, key_flat, &value_flat](int64 begin, int64 end) {
      for (int64 i = begin; i < end; ++i) {
        if (i >= key_flat.size()) {
          break;
        }
        table->insert_or_assign(key_flat(i), value_flat, value_dim_, i);
      }
    };
    auto& worker_threads = *context->device()->tensorflow_cpu_worker_threads();
    // Only use num_worker_threads when
    // TFRA_NUM_WORKER_THREADS_FOR_LOOKUP_TABLE_INSERT env var is set to k where
    // k > 0 and k <current number of tf cpu worker threads. Otherwise nothing
    // changes.
    int64 num_worker_threads = -1;
    Status status =
        ReadInt64FromEnvVar("TFRA_NUM_WORKER_THREADS_FOR_LOOKUP_TABLE_INSERT",
                            -1, &num_worker_threads);
    if (!status.ok()) {
      LOG(ERROR)
          << "Error parsing TFRA_NUM_WORKER_THREADS_FOR_LOOKUP_TABLE_INSERT: "
          << status;
    }
    if (num_worker_threads <= 0 ||
        num_worker_threads > worker_threads.num_threads) {
      num_worker_threads = worker_threads.num_threads;
    }
    int64 slices = static_cast<int64>(total / worker_threads.num_threads) + 1;
    Shard(num_worker_threads, worker_threads.workers, total, slices, shard);
  }

 private:
  const int64 value_dim_;
};

template <class K, class V>
class CuckooHashTableOfTensors final : public LookupInterface {
 public:
  CuckooHashTableOfTensors(OpKernelContext* ctx, OpKernel* kernel) {
    int64 env_var = 0;
    int64 init_size = 0;
    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(kernel->def(), "value_shape", &value_shape_));
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "init_size", &init_size));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(value_shape_),
        errors::InvalidArgument("Default value must be a vector, got shape ",
                                value_shape_.DebugString()));
    init_size_ = static_cast<size_t>(init_size);
    if (init_size_ == 0) {
      Status status = ReadInt64FromEnvVar("TF_HASHTABLE_INIT_SIZE",
                                          1024 * 8,  // 8192 KV pairs by default
                                          &env_var);
      if (!status.ok()) {
        LOG(ERROR) << "Error parsing TF_HASHTABLE_INIT_SIZE: " << status;
      }
      init_size_ = env_var;
    }
    runtime_dim_ = value_shape_.dim_size(0);
    cpu::CreateTable(init_size_, runtime_dim_, &table_);
  }

  ~CuckooHashTableOfTensors() { delete table_; }

  size_t size() const override { return table_->size(); }

  Status Find(OpKernelContext* ctx, const Tensor& key, Tensor* value,
              const Tensor& default_value) override {
    int64 value_dim = value_shape_.dim_size(0);

    LaunchTensorsFind<CPUDevice, K, V> launcher(value_dim);
    launcher.launch(ctx, table_, key, value, default_value);

    return Status::OK();
  }

  Status DoInsert(bool clear, OpKernelContext* ctx, const Tensor& keys,
                  const Tensor& values) {
    int64 value_dim = value_shape_.dim_size(0);

    if (clear) {
      table_->clear();
    }

    LaunchTensorsInsert<CPUDevice, K, V> launcher(value_dim);
    launcher.launch(ctx, table_, keys, values);

    return Status::OK();
  }

  Status Insert(OpKernelContext* ctx, const Tensor& keys,
                const Tensor& values) override {
    return DoInsert(false, ctx, keys, values);
  }

  Status Remove(OpKernelContext* ctx, const Tensor& keys) override {
    const auto key_flat = keys.flat<K>();

    // mutex_lock l(mu_);
    for (int64 i = 0; i < key_flat.size(); ++i) {
      table_->erase(tensorflow::lookup::SubtleMustCopyIfIntegral(key_flat(i)));
    }
    return Status::OK();
  }

  Status ImportValues(OpKernelContext* ctx, const Tensor& keys,
                      const Tensor& values) override {
    return DoInsert(true, ctx, keys, values);
  }

  Status ExportValues(OpKernelContext* ctx) override {
    int64 value_dim = value_shape_.dim_size(0);
    return table_->export_values(ctx, value_dim);
  }

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }

  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }

  TensorShape key_shape() const final { return TensorShape(); }

  TensorShape value_shape() const override { return value_shape_; }

  int64 MemoryUsed() const override {
    int64 ret = 0;
    ret = (int64)table_->size();
    return sizeof(CuckooHashTableOfTensors) + ret;
  }

 private:
  TensorShape value_shape_;
  size_t runtime_dim_;
  cpu::TableWrapperBase<K, V>* table_ = nullptr;
  size_t init_size_;
};

}  // namespace lookup

class HashTableOpKernel : public OpKernel {
 public:
  explicit HashTableOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx),
        expected_input_0_(ctx->input_type(0) == DT_RESOURCE ? DT_RESOURCE
                                                            : DT_STRING_REF) {}

 protected:
  Status LookupResource(OpKernelContext* ctx, const ResourceHandle& p,
                        LookupInterface** value) {
    return ctx->resource_manager()->Lookup<LookupInterface, false>(
        p.container(), p.name(), value);
  }

  Status GetTableHandle(StringPiece input_name, OpKernelContext* ctx,
                        string* container, string* table_handle) {
    {
      mutex* mu;
      TF_RETURN_IF_ERROR(ctx->input_ref_mutex(input_name, &mu));
      mutex_lock l(*mu);
      Tensor tensor;
      TF_RETURN_IF_ERROR(ctx->mutable_input(input_name, &tensor, true));
      if (tensor.NumElements() != 2) {
        return errors::InvalidArgument(
            "Lookup table handle must be scalar, but had shape: ",
            tensor.shape().DebugString());
      }
      auto h = tensor.flat<tstring>();
      *container = h(0);
      *table_handle = h(1);
    }
    return Status::OK();
  }

  Status GetResourceHashTable(StringPiece input_name, OpKernelContext* ctx,
                              LookupInterface** table) {
    const Tensor* handle_tensor;
    TF_RETURN_IF_ERROR(ctx->input(input_name, &handle_tensor));
    const ResourceHandle& handle = handle_tensor->scalar<ResourceHandle>()();
    return this->LookupResource(ctx, handle, table);
  }

  Status GetReferenceLookupTable(StringPiece input_name, OpKernelContext* ctx,
                                 LookupInterface** table) {
    string container;
    string table_handle;
    TF_RETURN_IF_ERROR(
        this->GetTableHandle(input_name, ctx, &container, &table_handle));
    return ctx->resource_manager()->Lookup(container, table_handle, table);
  }

  Status GetTable(OpKernelContext* ctx, LookupInterface** table) {
    if (expected_input_0_ == DT_RESOURCE) {
      return this->GetResourceHashTable("table_handle", ctx, table);
    } else {
      return this->GetReferenceLookupTable("table_handle", ctx, table);
    }
  }

  const DataType expected_input_0_;
};

class HashTableFindOp : public HashTableOpKernel {
 public:
  using HashTableOpKernel::HashTableOpKernel;

  void Compute(OpKernelContext* ctx) override {
    LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor& key = ctx->input(1);
    const Tensor& default_value = ctx->input(2);

    TensorShape output_shape = key.shape();
    output_shape.RemoveLastDims(table->key_shape().dims());
    output_shape.AppendShape(table->value_shape());
    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("values", output_shape, &out));

    OP_REQUIRES_OK(ctx, table->Find(ctx, key, out, default_value));
  }
};

// Table insert op.
class HashTableInsertOp : public HashTableOpKernel {
 public:
  using HashTableOpKernel::HashTableOpKernel;

  void Compute(OpKernelContext* ctx) override {
    LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                      table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForInsert(keys, values));

    int64 memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, table->Insert(ctx, keys, values));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

// Table remove op.
class HashTableRemoveOp : public HashTableOpKernel {
 public:
  using HashTableOpKernel::HashTableOpKernel;

  void Compute(OpKernelContext* ctx) override {
    LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& key = ctx->input(1);
    OP_REQUIRES_OK(ctx, table->CheckKeyTensorForRemove(key));

    int64 memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, table->Remove(ctx, key));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

// Op that returns the size of the given table.
class HashTableSizeOp : public HashTableOpKernel {
 public:
  using HashTableOpKernel::HashTableOpKernel;

  void Compute(OpKernelContext* ctx) override {
    LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("size", TensorShape({}), &out));
    out->flat<int64>().setConstant(table->size());
  }
};

// Op that outputs tensors of all keys and all values.
class HashTableExportOp : public HashTableOpKernel {
 public:
  using HashTableOpKernel::HashTableOpKernel;

  void Compute(OpKernelContext* ctx) override {
    LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, table->ExportValues(ctx));
  }
};

// Clear the table and insert data.
class HashTableImportOp : public HashTableOpKernel {
 public:
  using HashTableOpKernel::HashTableOpKernel;

  void Compute(OpKernelContext* ctx) override {
    LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                      table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& keys = ctx->input(1);
    const Tensor& values = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForImport(keys, values));

    int memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, table->ImportValues(ctx, keys, values));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(CuckooHashTableFind)).Device(DEVICE_CPU),
    HashTableFindOp);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(CuckooHashTableInsert)).Device(DEVICE_CPU),
    HashTableInsertOp);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(CuckooHashTableRemove)).Device(DEVICE_CPU),
    HashTableRemoveOp);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(CuckooHashTableSize)).Device(DEVICE_CPU),
    HashTableSizeOp);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(CuckooHashTableExport)).Device(DEVICE_CPU),
    HashTableExportOp);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(CuckooHashTableImport)).Device(DEVICE_CPU),
    HashTableImportOp);

// Register the CuckooMutableHashTableOfTensors op.
#define REGISTER_KERNEL(key_dtype, value_dtype)                             \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(PREFIX_OP_NAME(CuckooHashTableOfTensors))                        \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<key_dtype>("key_dtype")                           \
          .TypeConstraint<value_dtype>("value_dtype"),                      \
      HashTableOp<lookup::CuckooHashTableOfTensors<key_dtype, value_dtype>, \
                  key_dtype, value_dtype>)

REGISTER_KERNEL(int32, double);
REGISTER_KERNEL(int32, float);
REGISTER_KERNEL(int32, int32);
REGISTER_KERNEL(int64, double);
REGISTER_KERNEL(int64, float);
REGISTER_KERNEL(int64, int32);
REGISTER_KERNEL(int64, int64);
REGISTER_KERNEL(int64, tstring);
REGISTER_KERNEL(int64, int8);
REGISTER_KERNEL(int64, Eigen::half);
REGISTER_KERNEL(tstring, bool);
REGISTER_KERNEL(tstring, double);
REGISTER_KERNEL(tstring, float);
REGISTER_KERNEL(tstring, int32);
REGISTER_KERNEL(tstring, int64);
REGISTER_KERNEL(tstring, int8);
REGISTER_KERNEL(tstring, Eigen::half);

#undef REGISTER_KERNEL

}  // namespace recommenders_addons
}  // namespace tensorflow
