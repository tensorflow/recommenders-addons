#include "lookup_table.hpp"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/utils.h"

#include "tensorflow/core/kernels/lookup_table_op.h"

namespace tensorflow {
namespace recommenders_addons {
namespace lookup_table {

using namespace tensorflow::lookup;

class LookupTableOpKernel : public OpKernel {
 public:
  explicit LookupTableOpKernel(OpKernelConstruction *ctx)
      : OpKernel(ctx),
        expected_input_0_(ctx->input_type(0) == tensorflow::DataType::DT_RESOURCE ? tensorflow::DataType::DT_RESOURCE
                                                            : tensorflow::DataType::DT_STRING_REF) {}

 protected:
  Status LookupResource(OpKernelContext *ctx, const ResourceHandle &p,
                        LookupInterface **value) {
    return ctx->resource_manager()->Lookup<LookupInterface, false>(
        p.container(), p.name(), value);
  }

  Status GetTableHandle(StringPiece input_name, OpKernelContext *ctx,
                        string *container, string *table_handle) {
    {
      mutex *mu;
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

  Status GetResourceLookupTable(StringPiece input_name, OpKernelContext *ctx,
                              LookupInterface **table) {
    const Tensor *handle_tensor;
    TF_RETURN_IF_ERROR(ctx->input(input_name, &handle_tensor));
    const ResourceHandle &handle = handle_tensor->scalar<ResourceHandle>()();
    return this->LookupResource(ctx, handle, table);
  }

  Status GetReferenceLookupTable(StringPiece input_name, OpKernelContext *ctx,
                                 LookupInterface **table) {
    string container;
    string table_handle;
    TF_RETURN_IF_ERROR(
        this->GetTableHandle(input_name, ctx, &container, &table_handle));
    return ctx->resource_manager()->Lookup(container, table_handle, table);
  }

  Status GetTable(OpKernelContext *ctx, LookupInterface **table) {
    if (expected_input_0_ == tensorflow::DataType::DT_RESOURCE) {
      return this->GetResourceLookupTable("table_handle", ctx, table);
    } else {
      return this->GetReferenceLookupTable("table_handle", ctx, table);
    }
  }

  const DataType expected_input_0_;
};  // class LookupTableOpKernel


// Table find op 
class LookupTableFindOp : public LookupTableOpKernel {
 public:
  using LookupTableOpKernel::LookupTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor &key = ctx->input(1);
    const Tensor &default_value = ctx->input(2);

    LOG(INFO) << "Find key: " << key.DebugString();
    LOG(INFO) << "Find default value: " << default_value.DebugString();

    TensorShape output_shape = key.shape();
    output_shape.RemoveLastDims(table->key_shape().dims());
    output_shape.AppendShape(table->value_shape());
    Tensor *out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("values", output_shape, &out));

    OP_REQUIRES_OK(ctx, table->Find(ctx, key, out, default_value));
  }
};

// Table find op with return exists tensor.
template <class K, class V>
class LookupTableFindWithExistsOp : public LookupTableOpKernel {
 public:
  using LookupTableOpKernel::LookupTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    lookup_table::LookupTableOfTensors<K, V> *lookup_table =
        dynamic_cast<lookup_table::LookupTableOfTensors<K, V> *>(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {table->value_dtype(), tensorflow::DataType::DT_BOOL};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor &key = ctx->input(1);
    const Tensor &default_value = ctx->input(2);

    TensorShape output_shape = key.shape();
    output_shape.RemoveLastDims(table->key_shape().dims());
    output_shape.AppendShape(table->value_shape());

    Tensor *values;
    Tensor *exists;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("values", output_shape, &values));
    OP_REQUIRES_OK(ctx, ctx->allocate_output("exists", key.shape(), &exists));

    OP_REQUIRES_OK(ctx, lookup_table->FindWithExists(ctx, key, values,
                                                    default_value, *exists));
  }
};

// Table insert op.
class LookupTableInsertOp : public LookupTableOpKernel {
 public:
  using LookupTableOpKernel::LookupTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                      table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor &keys = ctx->input(1);
    const Tensor &values = ctx->input(2);

    LOG(INFO) << "insert Compute keys shape string: " << keys.shape().DebugString();
    LOG(INFO) << "insert Compute keys debug string: " << keys.DebugString();
    LOG(INFO) << "insert Compute values debug string: " << values.DebugString();
    LOG(INFO) << "insert Compute values shape string: " << values.shape().DebugString();


    OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForInsert(keys, values));

    int64_t memory_used_before = 0;
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

// Table accum op.
template <class K, class V>
class LookupTableAccumOp : public LookupTableOpKernel {
 public:
  using LookupTableOpKernel::LookupTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    LookupTableOfTensors<K, V> *lookup_table = (LookupTableOfTensors<K, V> *)table;

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                      table->value_dtype(),
                                      tensorflow::DataTypeToEnum<bool>::v()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor &keys = ctx->input(1);
    const Tensor &values_or_deltas = ctx->input(2);
    const Tensor &exists = ctx->input(3);

    LOG(INFO) << "Accum keys: " << keys.DebugString();
    LOG(INFO) << "Accum values or deltas: " << values_or_deltas.DebugString();
    LOG(INFO) << "Accum  exists: " << exists.DebugString();

    auto akeys = keys.flat<K>();
    for (auto i =0; i<akeys.size(); i++) {
      LOG(INFO) << "akeys i: " << i << " " << akeys(i);
    }

    auto avalues_or_deltas = values_or_deltas.flat<V>();
    for (auto i =0; i<avalues_or_deltas.size(); i++) {
      LOG(INFO) << "avalues_or_deltas i: " << i << " " << avalues_or_deltas(i);
    }

    OP_REQUIRES(ctx, (values_or_deltas.dtype() != tensorflow::DataTypeToEnum<tstring>::v()),
                errors::InvalidArgument(
                    "AccumOP is not supporting tstring value type!"));
    OP_REQUIRES_OK(
        ctx, table->CheckKeyAndValueTensorsForInsert(keys, values_or_deltas));

    int64_t memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx,
                   lookup_table->Accum(ctx, keys, values_or_deltas, exists));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

// Table remove op.
class LookupTableRemoveOp : public LookupTableOpKernel {
 public:
  using LookupTableOpKernel::LookupTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor &key = ctx->input(1);
    OP_REQUIRES_OK(ctx, table->CheckKeyTensorForRemove(key));

    LOG(INFO) << "remove key: " << key.DebugString();

    int64_t memory_used_before = 0;
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

// Table clear op.
template <class K, class V>
class LookupTableClearOp : public LookupTableOpKernel {
 public:
  using LookupTableOpKernel::LookupTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    lookup_table::LookupTableOfTensors<K, V> *lookup_table =
        dynamic_cast<lookup_table::LookupTableOfTensors<K, V> *>(table);

    int64_t memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, lookup_table->Clear(ctx));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

// Op that returns the size of the given table.
class LookupTableSizeOp : public LookupTableOpKernel {
 public:
  using LookupTableOpKernel::LookupTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    Tensor *out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("size", TensorShape({}), &out));
    out->flat<int64_t>().setConstant(table->size());
  }
};

// Op that outputs tensors of all keys and all values.
class LookupTableExportOp : public LookupTableOpKernel {
 public:
  using LookupTableOpKernel::LookupTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, table->ExportValues(ctx));
  }
};

// Op that export all keys and values to FileSystem.
template <class K, class V>
class LookupTableSaveToFileSystemOp : public LookupTableOpKernel {
 public:
  explicit LookupTableSaveToFileSystemOp(OpKernelConstruction *ctx)
      : LookupTableOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dirpath_env", &dirpath_env_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("append_to_file", &append_to_file_));
    int64 signed_buffer_size = 0;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("buffer_size", &signed_buffer_size));
    buffer_size_ = static_cast<size_t>(signed_buffer_size);
  }

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    string dirpath;
    TF_CHECK_OK(tensorflow::ReadStringFromEnvVar(dirpath_env_, "NotFound", &dirpath));
    if (dirpath != "NotFound") {
      LOG(INFO) << "Read TFRA key/value file directory path from the "
                   "environment variable "
                << dirpath_env_ << " successfully. Saving directory path is "
                << dirpath;
    } else {
      const Tensor &dir_tensor = ctx->input(1);
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(dir_tensor.shape()),
                  errors::InvalidArgument("directory path must be scalar."));
      dirpath = string(dir_tensor.scalar<tstring>()().data());
    }

    const Tensor &fname_tensor = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(fname_tensor.shape()),
                errors::InvalidArgument("file name must be scalar."));
    string file_name = string(fname_tensor.scalar<tstring>()().data());

    LookupTableOfTensors<K, V> *lookup_table = (LookupTableOfTensors<K, V> *)table;
    OP_REQUIRES_OK(
        ctx, lookup_table->SaveToFileSystem(ctx, dirpath, file_name,
                                           buffer_size_, append_to_file_));
  }

 private:
  string dirpath_env_;
  bool append_to_file_;
  size_t buffer_size_;
};

// Insert data.
class LookupTableImportOp : public LookupTableOpKernel {
 public:
  using LookupTableOpKernel::LookupTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                      table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor &keys = ctx->input(1);
    const Tensor &values = ctx->input(2);
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

// Insert data from FileSystem.
template <class K, class V>
class LookupTableLoadFromFileSystemOp : public LookupTableOpKernel {
 public:
  explicit LookupTableLoadFromFileSystemOp(OpKernelConstruction *ctx)
      : LookupTableOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dirpath_env", &dirpath_env_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("load_entire_dir", &load_entire_dir_));
    int64 signed_buffer_size = 0;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("buffer_size", &signed_buffer_size));
    buffer_size_ = static_cast<size_t>(signed_buffer_size);
  }

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    string dirpath;
    TF_CHECK_OK(tensorflow::ReadStringFromEnvVar(dirpath_env_, "NotFound", &dirpath));
    if (dirpath != "NotFound") {
      LOG(INFO) << "Read TFRA key/value file directory path from the "
                   "environment variable "
                << dirpath_env_ << " successfully. Saving directory path is "
                << dirpath;
    } else {
      const Tensor &dir_tensor = ctx->input(1);
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(dir_tensor.shape()),
                  errors::InvalidArgument("directory path must be scalar."));
      dirpath = string(dir_tensor.scalar<tstring>()().data());
    }

    const Tensor &fname_tensor = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(fname_tensor.shape()),
                errors::InvalidArgument("file name must be scalar."));
    string file_name = string(fname_tensor.scalar<tstring>()().data());

    LookupTableOfTensors<K, V> *lookup_table = (LookupTableOfTensors<K, V> *)table;
    OP_REQUIRES_OK(
        ctx, lookup_table->LoadFromFileSystem(ctx, dirpath, file_name,
                                             buffer_size_, load_entire_dir_));
  }

 private:
  string dirpath_env_;
  bool load_entire_dir_;
  size_t buffer_size_;
};

REGISTER_KERNEL_BUILDER(Name(PREFIX_OP_NAME(LookupTableFind)).Device(DEVICE_CPU),
                        LookupTableFindOp);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(LookupTableInsert)).Device(DEVICE_CPU),
    LookupTableInsertOp);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(LookupTableRemove)).Device(DEVICE_CPU),
    LookupTableRemoveOp);
REGISTER_KERNEL_BUILDER(Name(PREFIX_OP_NAME(LookupTableSize)).Device(DEVICE_CPU),
                        LookupTableSizeOp);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(LookupTableExport)).Device(DEVICE_CPU),
    LookupTableExportOp);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(LookupTableImport)).Device(DEVICE_CPU),
    LookupTableImportOp);

// Register the custom op.
#define REGISTER_LOOKUP_TABLE_KERNEL(key_dtype, value_dtype)                \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(PREFIX_OP_NAME(LookupTableOfTensors))                            \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<key_dtype>("key_dtype")                           \
          .TypeConstraint<value_dtype>("value_dtype"),                      \
  LookupTableOp<lookup_table::LookupTableOfTensors<key_dtype, value_dtype>, \
                  key_dtype, value_dtype>);                                 \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(PREFIX_OP_NAME(LookupTableClear))                                \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<key_dtype>("key_dtype")                           \
          .TypeConstraint<value_dtype>("value_dtype"),                      \
      lookup_table::LookupTableClearOp<key_dtype, value_dtype>);            \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(PREFIX_OP_NAME(LookupTableAccum))                                \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<key_dtype>("key_dtype")                           \
          .TypeConstraint<value_dtype>("value_dtype"),                      \
      lookup_table::LookupTableAccumOp<key_dtype, value_dtype>);            \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(PREFIX_OP_NAME(LookupTableFindWithExists))                       \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<key_dtype>("Tin")                                 \
          .TypeConstraint<value_dtype>("Tout"),                             \
      lookup_table::LookupTableFindWithExistsOp<key_dtype, value_dtype>);   \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(PREFIX_OP_NAME(LookupTableSaveToFileSystem))                     \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<key_dtype>("key_dtype")                           \
          .TypeConstraint<value_dtype>("value_dtype"),                      \
      lookup_table::LookupTableSaveToFileSystemOp<key_dtype, value_dtype>); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(PREFIX_OP_NAME(LookupTableLoadFromFileSystem))                   \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<key_dtype>("key_dtype")                           \
          .TypeConstraint<value_dtype>("value_dtype"),                      \
      lookup_table::LookupTableLoadFromFileSystemOp<key_dtype, value_dtype>);


REGISTER_LOOKUP_TABLE_KERNEL(int32, double);
REGISTER_LOOKUP_TABLE_KERNEL(int32, float);
REGISTER_LOOKUP_TABLE_KERNEL(int32, int32);
REGISTER_LOOKUP_TABLE_KERNEL(int64_t, double);
REGISTER_LOOKUP_TABLE_KERNEL(int64_t, float);
REGISTER_LOOKUP_TABLE_KERNEL(int64_t, int32);
REGISTER_LOOKUP_TABLE_KERNEL(int64_t, int64_t);
// REGISTER_LOOKUP_TABLE_KERNEL(int64_t, tstring);

// REGISTER_LOOKUP_TABLE_KERNEL(int64_t, tstring);
REGISTER_LOOKUP_TABLE_KERNEL(int64_t, int8);
// REGISTER_LOOKUP_TABLE_KERNEL(int64_t, Eigen::half);
// REGISTER_LOOKUP_TABLE_KERNEL(tstring, bool);
// REGISTER_LOOKUP_TABLE_KERNEL(tstring, double);
// REGISTER_LOOKUP_TABLE_KERNEL(tstring, float);
// REGISTER_LOOKUP_TABLE_KERNEL(tstring, int32);
// REGISTER_LOOKUP_TABLE_KERNEL(tstring, int64_t);
// REGISTER_LOOKUP_TABLE_KERNEL(tstring, int8);
// REGISTER_LOOKUP_TABLE_KERNEL(tstring, Eigen::half);

#undef REGISTER_LOOKUP_TABLE_KERNEL

}   // lookup_table_cpu
}   // recommenders_addons
}   // tensorflow