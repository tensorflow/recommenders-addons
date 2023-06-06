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
#include "lookup_table.hpp"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/kernels/lookup_impl/lookup_table_op_gpu.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/utils.h"

#define EIGEN_USE_GPU

#include <cuda_runtime.h>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/stream_executor/stream.h"

namespace tensorflow {
namespace recommenders_addons {
namespace lookup_table {

template <class Container, class key_dtype, class value_dtype>
class LookupTableGpuOp : public OpKernel {
 public:
  // ctx is not owned by this class.
  explicit LookupTableGpuOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), table_set_(false) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("use_node_name_sharing", &use_node_name_sharing_));
  }

  // ctx is not owned by this function.
  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);

    if (!table_set_) {
      AllocatorAttributes attr;
      attr.set_gpu_compatible(true);
      attr.set_on_host(true);
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(tensorflow::DT_STRING,
                                  tensorflow::TensorShape({2}), &table_, attr));

      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                      use_node_name_sharing_));
    }

    auto creator =
        [ctx, this](lookup::LookupInterface** ret)
            TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
              lookup::LookupInterface* container = new Container(ctx, this);
              if (!ctx->status().ok()) {
                container->Unref();
                return ctx->status();
              }
              if (ctx->track_allocations()) {
                ctx->record_persistent_memory_allocation(
                    container->MemoryUsed() + table_.AllocatedBytes());
              }
              *ret = container;
              return Status::OK();
            };

    lookup::LookupInterface* table = nullptr;
    OP_REQUIRES_OK(ctx,
                   cinfo_.resource_manager()
                       ->template LookupOrCreate<lookup::LookupInterface>(
                           cinfo_.container(), cinfo_.name(), &table, creator));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, lookup::CheckTableDataTypes(
                            *table, tensorflow::DataTypeToEnum<key_dtype>::v(),
                            tensorflow::DataTypeToEnum<value_dtype>::v(), cinfo_.name()));

    if (ctx->expected_output_dtype(0) == DT_RESOURCE) {
      Tensor* handle;
#if GOOGLE_CUDA
      AllocatorAttributes attr;
      attr.set_gpu_compatible(true);
      attr.set_on_host(true);
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(0, TensorShape({}), &handle, attr));
#else
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle));
#endif
      handle->scalar<ResourceHandle>()() =
          MakeResourceHandle<lookup::LookupInterface>(ctx, cinfo_.container(),
                                                      cinfo_.name());
    } else {
      if (!table_set_) {
        auto h = table_.template flat<tstring>();
        h(0) = cinfo_.container();
        h(1) = cinfo_.name();
      }
      ctx->set_output_ref(0, &mu_, &table_);
    }
    table_set_ = true;
  }

  ~LookupTableGpuOp() override {
    // If the table object was not shared, delete it.
    if (table_set_ && cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->template Delete<lookup::LookupInterface>(cinfo_.container(),
                                                          cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }

 private:
  mutex mu_;
  Tensor table_ TF_GUARDED_BY(mu_);
  bool table_set_ TF_GUARDED_BY(mu_);
  ContainerInfo cinfo_;
  bool use_node_name_sharing_;

  TF_DISALLOW_COPY_AND_ASSIGN(LookupTableGpuOp);
};  // class LookupTableGpuOp


// Table lookup op. Perform the lookup operation on the given table.
class LookupTableFindGpuOp : public OpKernel {
 public:
  explicit LookupTableFindGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    // Input 0 could be a STRING_REF or a RESOURCE
    DataType expected_input_0 = DT_RESOURCE;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor& keys = ctx->input(1);
    const Tensor& default_values = ctx->input(2);

    TensorShape output_shape = keys.shape();
    output_shape.RemoveLastDims(table->key_shape().dims());
    output_shape.AppendShape(table->value_shape());
    Tensor* out;
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("values", output_shape, &out, attr));

    OP_REQUIRES_OK(ctx, table->Find(ctx, keys, out, default_values));
  }
};

REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(LookupTableFind)).Device(DEVICE_GPU),
    LookupTableFindGpuOp);

// Table lookup op. Perform the lookup operation on the given table.

template <class K, class V>
class LookupTableFindWithExistsGpuOp : public OpKernel {
 public:
  explicit LookupTableFindWithExistsGpuOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    lookup_table::LookupTableOfTensors<K, V>* lookup_table =
        (lookup_table::LookupTableOfTensors<K, V>*)table;

    // Input 0 could be a STRING_REF or a RESOURCE
    DataType expected_input_0 = DT_RESOURCE;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {table->value_dtype(), DT_BOOL};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor& keys = ctx->input(1);
    const Tensor& default_values = ctx->input(2);

    TensorShape output_shape = keys.shape();
    output_shape.RemoveLastDims(table->key_shape().dims());
    output_shape.AppendShape(table->value_shape());
    Tensor* values;
    Tensor* exists;
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("values", output_shape, &values, attr));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("exists", keys.shape(), &exists, attr));

    OP_REQUIRES_OK(ctx, lookup_table->FindWithExists(ctx, keys, values,
                                                     default_values, exists));
  }
};

// Table insert op.
class LookupTableInsertGpuOp : public OpKernel {
 public:
  explicit LookupTableInsertGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 = DT_RESOURCE;
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
    Name(PREFIX_OP_NAME(LookupTableInsert)).Device(DEVICE_GPU),
    LookupTableInsertGpuOp);

// Table accum op.
template <class K, class V>
class LookupTableAccumGpuOp : public OpKernel {
 public:
  explicit LookupTableAccumGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);
    lookup_table::LookupTableOfTensors<K, V>* lookup_table =
        (lookup_table::LookupTableOfTensors<K, V>*)table;

    DataType expected_input_0 = DT_RESOURCE;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype(),
                                      tensorflow::DataTypeToEnum<bool>::v()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& keys = ctx->input(1);
    const Tensor& values_or_deltas = ctx->input(2);
    const Tensor& exists = ctx->input(3);
    OP_REQUIRES_OK(
        ctx, table->CheckKeyAndValueTensorsForInsert(keys, values_or_deltas));
    OP_REQUIRES_OK(ctx,
                   lookup_table->Accum(ctx, keys, values_or_deltas, exists));
  }
};

// Table remove op.
class LookupTableRemoveGpuOp : public OpKernel {
 public:
  explicit LookupTableRemoveGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 = DT_RESOURCE;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& key = ctx->input(1);
    OP_REQUIRES_OK(ctx, table->CheckKeyTensorForRemove(key));
    OP_REQUIRES_OK(ctx, table->Remove(ctx, key));
  }
};

REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(LookupTableRemove)).Device(DEVICE_GPU),
    LookupTableRemoveGpuOp);

// Table clear op.
template <class K, class V>
class LookupTableClearGpuOp : public OpKernel {
 public:
  explicit LookupTableClearGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);
    lookup_table::LookupTableOfTensors<K, V>* lookup_table =
        (lookup_table::LookupTableOfTensors<K, V>*)table;
    OP_REQUIRES_OK(ctx, lookup_table->Clear(ctx));
  }
};

// Op that returns the size of the given table.
class LookupTableSizeGpuOp : public OpKernel {
 public:
  explicit LookupTableSizeGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

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
    Name(PREFIX_OP_NAME(LookupTableSize)).Device(DEVICE_GPU),
    LookupTableSizeGpuOp);

// Op that outputs tensors of all keys and all values.
class LookupTableExportGpuOp : public OpKernel {
 public:
  explicit LookupTableExportGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, table->ExportValues(ctx));
  }
};

REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(LookupTableExport)).Device(DEVICE_GPU),
    LookupTableExportGpuOp);

// Op that export all keys and values to FileSystem.
template <class K, class V>
class LookupTableSaveToFileSystemGpuOp : public OpKernel {
 public:
  explicit LookupTableSaveToFileSystemGpuOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dirpath_env", &dirpath_env_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("append_to_file", &append_to_file_));
    int64 signed_buffer_size = 0;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("buffer_size", &signed_buffer_size));
    buffer_size_ = static_cast<size_t>(signed_buffer_size);
  }

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    string dirpath;
    TF_CHECK_OK(ReadStringFromEnvVar(dirpath_env_, "NotFound", &dirpath));
    if (dirpath != "NotFound") {
      LOG(INFO) << "Read TFRA key/value file directory path from the "
                   "environment variable "
                << dirpath_env_ << " successfully. Saving directory path is "
                << dirpath;
    } else {
      const Tensor& dir_tensor = ctx->input(1);
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(dir_tensor.shape()),
                  errors::InvalidArgument("directory path must be scalar."));
      dirpath = string(dir_tensor.scalar<tstring>()().data());
    }

    const Tensor& fname_tensor = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(fname_tensor.shape()),
                errors::InvalidArgument("file name must be scalar."));
    string file_name = string(fname_tensor.scalar<tstring>()().data());

    lookup_table::LookupTableOfTensors<K, V>* lookup_table =
        (lookup_table::LookupTableOfTensors<K, V>*)table;
    OP_REQUIRES_OK(
        ctx, lookup_table->SaveToFileSystem(ctx, dirpath, file_name,
                                            buffer_size_, append_to_file_));
  }

 private:
  string dirpath_env_;
  bool append_to_file_;
  size_t buffer_size_;
};

// Clear the table and insert data.
class LookupTableImportGpuOp : public OpKernel {
 public:
  explicit LookupTableImportGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    DataType expected_input_0 = DT_RESOURCE;
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
    Name(PREFIX_OP_NAME(LookupTableImport)).Device(DEVICE_GPU),
    LookupTableImportGpuOp);

// Clear the table and insert data from FileSystem.
template <class K, class V>
class LookupTableLoadFromFileSystemGpuOp : public OpKernel {
 public:
  explicit LookupTableLoadFromFileSystemGpuOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dirpath_env", &dirpath_env_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("load_entire_dir", &load_entire_dir_));
    int64 signed_buffer_size = 0;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("buffer_size", &signed_buffer_size));
    buffer_size_ = static_cast<size_t>(signed_buffer_size);
  }

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    string dirpath;
    TF_CHECK_OK(ReadStringFromEnvVar(dirpath_env_, "NotFound", &dirpath));
    if (dirpath != "NotFound") {
      LOG(INFO) << "Read TFRA key/value file directory path from the "
                   "environment variable "
                << dirpath_env_ << " successfully. Saving directory path is "
                << dirpath;
    } else {
      const Tensor& dir_tensor = ctx->input(1);
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(dir_tensor.shape()),
                  errors::InvalidArgument("directory path must be scalar."));
      dirpath = string(dir_tensor.scalar<tstring>()().data());
    }

    const Tensor& fname_tensor = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(fname_tensor.shape()),
                errors::InvalidArgument("file name must be scalar."));
    string file_name = string(fname_tensor.scalar<tstring>()().data());

    lookup_table::LookupTableOfTensors<K, V>* lookup_table =
        (lookup_table::LookupTableOfTensors<K, V>*)table;
    OP_REQUIRES_OK(
        ctx, lookup_table->LoadFromFileSystem(ctx, dirpath, file_name,
                                              buffer_size_, load_entire_dir_));
  }

 private:
  string dirpath_env_;
  bool load_entire_dir_;
  size_t buffer_size_;
};

#define REGISTER_LOOKUP_TABLE_KERNEL(key_dtype, value_dtype)                          \
  REGISTER_KERNEL_BUILDER(                                                            \
      Name(PREFIX_OP_NAME(LookupTableOfTensors))                                      \
          .Device(DEVICE_GPU)                                                         \
          .TypeConstraint<key_dtype>("key_dtype")                                     \
          .TypeConstraint<value_dtype>("value_dtype"),                                \
      LookupTableGpuOp<lookup_table::LookupTableOfTensors<key_dtype, value_dtype>,    \
                  key_dtype, value_dtype>);                                           \
  REGISTER_KERNEL_BUILDER(Name(PREFIX_OP_NAME(LookupTableClear))                      \
                              .Device(DEVICE_GPU)                                     \
                              .TypeConstraint<key_dtype>("key_dtype")                 \
                              .TypeConstraint<value_dtype>("value_dtype"),            \
                          LookupTableClearGpuOp<key_dtype, value_dtype>)              \
  REGISTER_KERNEL_BUILDER(Name(PREFIX_OP_NAME(LookupTableAccum))                      \
                              .Device(DEVICE_GPU)                                     \
                              .TypeConstraint<key_dtype>("key_dtype")                 \
                              .TypeConstraint<value_dtype>("value_dtype"),            \
                          LookupTableAccumGpuOp<key_dtype, value_dtype>)              \
  REGISTER_KERNEL_BUILDER(                                                            \
      Name(PREFIX_OP_NAME(LookupTableFindWithExists))                                 \
          .Device(DEVICE_GPU)                                                         \
          .TypeConstraint<key_dtype>("Tin")                                           \
          .TypeConstraint<value_dtype>("Tout"),                                       \
      LookupTableFindWithExistsGpuOp<key_dtype, value_dtype>)                         \
  REGISTER_KERNEL_BUILDER(                                                            \
      Name(PREFIX_OP_NAME(LookupTableSaveToFileSystem))                               \
          .Device(DEVICE_GPU)                                                         \
          .TypeConstraint<key_dtype>("key_dtype")                                     \
          .TypeConstraint<value_dtype>("value_dtype"),                                \
      LookupTableSaveToFileSystemGpuOp<key_dtype, value_dtype>);                      \
  REGISTER_KERNEL_BUILDER(                                                            \
      Name(PREFIX_OP_NAME(LookupTableLoadFromFileSystem))                             \
          .Device(DEVICE_GPU)                                                         \
          .TypeConstraint<key_dtype>("key_dtype")                                     \
          .TypeConstraint<value_dtype>("value_dtype"),                                \
      LookupTableLoadFromFileSystemGpuOp<key_dtype, value_dtype>);

REGISTER_LOOKUP_TABLE_KERNEL(int64, float);
// REGISTER_LOOKUP_TABLE_KERNEL(int64, Eigen::half);
REGISTER_LOOKUP_TABLE_KERNEL(int64, int64);
REGISTER_LOOKUP_TABLE_KERNEL(int64, int32);
REGISTER_LOOKUP_TABLE_KERNEL(int64, int8);
REGISTER_LOOKUP_TABLE_KERNEL(int32, float);

#undef REGISTER_LOOKUP_TABLE_KERNELb

}  // namespace lookup
}  // namespace recommenders_addons
}  // namespace tensorflow
#endif
