/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow_recommenders_addons/dynamic_embedding/core/kernels/lookup_impl/lookup_table_op_hkv.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/utils.h"

#define EIGEN_USE_GPU

#include <cuda_runtime.h>
#include <stdlib.h>

#include <array>
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
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/stream_executor/stream.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace recommenders_addons {
namespace lookup {

constexpr size_t kDefaultGpuInitCapacity = 1024 * 1024;

using tensorflow::OpKernelContext;
using tensorflow::lookup::LookupInterface;

template <class K, class V>
class HkvHashTableOfTensorsGpu final : public LookupInterface {
 private:
  std::unique_ptr<nv::merlin::BaseAllocator> allocator_ptr_;

 public:
  HkvHashTableOfTensorsGpu(OpKernelContext* ctx, OpKernel* kernel) {
    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(kernel->def(), "value_shape", &value_shape_));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(value_shape_),
        errors::InvalidArgument("Default value must be a vector, got shape ",
                                value_shape_.DebugString()));
    runtime_dim_ = value_shape_.dim_size(0);

    gpu::TableWrapperInitOptions options;

    int64 init_capacity_i64 = 0;
    int64 max_capacity_i64 = 0;
    int64 max_hbm_for_vectors_i64 = 0;
    OP_REQUIRES_OK(
        ctx, GetNodeAttr(kernel->def(), "init_capacity", &init_capacity_i64));
    OP_REQUIRES_OK(
        ctx, GetNodeAttr(kernel->def(), "max_capacity", &max_capacity_i64));
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "max_hbm_for_vectors",
                                    &max_hbm_for_vectors_i64));
    OP_REQUIRES(
        ctx, (max_hbm_for_vectors_i64 >= 0),
        errors::InvalidArgument("params max_hbm_for_vectors less than 0"));

    options.init_capacity = static_cast<size_t>(init_capacity_i64);
    options.max_capacity = static_cast<size_t>(max_capacity_i64);
    options.max_hbm_for_vectors = static_cast<size_t>(max_hbm_for_vectors_i64);

    if (options.max_capacity == 0) {
      char* env_max_capacity_str =
          std::getenv("TFRA_GPU_HASHTABLE_UPLIMIT_SIZE");
      OP_REQUIRES(ctx, (env_max_capacity_str != nullptr),
                  errors::InvalidArgument(
                      "max_capaicty=0 and TFRA_GPU_HASHTABLE_UPLIMIT_SIZE not "
                      "set is not valid."));
      options.max_capacity =
          static_cast<size_t>(std::atoll(env_max_capacity_str));
      LOG(WARNING) << "GPU table max capacity was not set in attribute, get "
                   << options.max_capacity
                   << " from env TFRA_GPU_HASHTABLE_UPLIMIT_SIZE.";
    }
    if (options.init_capacity == 0) {
      options.init_capacity = kDefaultGpuInitCapacity;
      LOG(WARNING)
          << "GPU table init capacity was not set in attribute, use default"
          << kDefaultGpuInitCapacity;
    }
    if (options.max_capacity < options.init_capacity) {
      LOG(WARNING) << "GPU table max_capacity < init_capacity, ("
                   << options.max_capacity << "/" << options.init_capacity
                   << "). Reset to " << options.init_capacity;
      options.max_capacity = options.init_capacity;
    }

    if (table_) {
      return;
    }
    allocator_ptr_ = std::make_unique<gpu::TFOrDefaultAllocator>(ctx);
    OP_REQUIRES_OK(ctx,
                   this->CreateTable(options, allocator_ptr_.get(), &table_));
    OP_REQUIRES(ctx, (table_ != nullptr),
                errors::InvalidArgument("HashTable on GPU is created failed!"));
    LOG(INFO) << "GPU table max capacity was created on max_capacity: "
              << options.max_capacity
              << ", and init capacity: " << options.init_capacity
              << " with K=" << std::type_index(typeid(K)).name()
              << ", V=" << std::type_index(typeid(V)).name();
  }

  ~HkvHashTableOfTensorsGpu() {
    mutex_lock l(mu_);
    if (table_) {
      delete table_;
      table_ = nullptr;
    }
  }

  Status CreateTable(gpu::TableWrapperInitOptions& options,
                     nv::merlin::BaseAllocator* allocator,
                     gpu::TableWrapper<K, V>** pptable) {
    return gpu::CreateTableImpl(pptable, options, allocator, runtime_dim_);
  }

  size_t size() const override {
    tf_shared_lock l(mu_);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    size_t retv = table_->get_size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return retv;
  }

  void size_i64(OpKernelContext* ctx, int64* s) {
    tf_shared_lock l(mu_);
    auto stream = ctx->eigen_device<GPUDevice>().stream();
    int64 hret = static_cast<int64>(table_->get_size(stream));
    CUDA_CHECK(cudaMemcpyAsync(s, &hret, sizeof(int64), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  Status Find(OpKernelContext* ctx, const Tensor& d_keys, Tensor* value,
              const Tensor& default_value) override {
    size_t len = d_keys.flat<K>().size();
    bool* d_status;

    auto value_flat = value->flat_inner_dims<V, 2>();
    const auto default_flat = default_value.flat<V>();
    int64 total = value_flat.size();
    int64 default_total = default_flat.size();
    bool is_full_default = (total == default_total);

    auto stream = ctx->eigen_device<GPUDevice>().stream();

    if (len > 0) {
      size_t default_value_num =
          is_full_default ? default_value.shape().dim_size(0) : 1;
      CUDA_CHECK(cudaMallocAsync(&d_status, sizeof(bool) * len, stream));
      CUDA_CHECK(cudaMemsetAsync(d_status, 0, sizeof(bool) * len, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      {
        tf_shared_lock l(mu_);
        try {
          table_->get((const K*)d_keys.tensor_data().data(),
                      (V*)(value->tensor_data().data()), d_status, len,
                      (V*)(default_value.tensor_data().data()), stream,
                      is_full_default);
          CUDA_CHECK(cudaStreamSynchronize(stream));
        } catch (std::runtime_error& e) {
          return Status(tensorflow::error::INTERNAL, e.what());
        }
      }
      CUDA_CHECK(cudaFreeAsync(d_status, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    return Status::OK();
  }

  Status FindWithExists(OpKernelContext* ctx, const Tensor& d_keys,
                        Tensor* value, const Tensor& default_value,
                        Tensor* exists) {
    size_t len = d_keys.flat<K>().size();

    auto value_flat = value->flat_inner_dims<V, 2>();
    const auto default_flat = default_value.flat<V>();
    int64 total = value_flat.size();
    int64 default_total = default_flat.size();
    bool is_full_default = (total == default_total);

    auto stream = ctx->eigen_device<GPUDevice>().stream();

    if (len > 0) {
      size_t default_value_num =
          is_full_default ? default_value.shape().dim_size(0) : 1;
      {
        tf_shared_lock l(mu_);
        try {
          table_->get((const K*)d_keys.tensor_data().data(),
                      (V*)(value->tensor_data().data()),
                      (bool*)exists->tensor_data().data(), len,
                      (V*)(default_value.tensor_data().data()), stream,
                      is_full_default);
        } catch (std::runtime_error& e) {
          return Status(tensorflow::error::INTERNAL, e.what());
        }
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    return Status::OK();
  }

  Status Insert(OpKernelContext* ctx, const Tensor& keys,
                const Tensor& values) override {
    size_t len = keys.flat<K>().size();
    auto stream = ctx->eigen_device<GPUDevice>().stream();
    {
      mutex_lock l(mu_);
      try {
        table_->upsert((const K*)keys.tensor_data().data(),
                       (const V*)(values.tensor_data().data()), len, stream);
      } catch (std::runtime_error& e) {
        return Status(tensorflow::error::INTERNAL, e.what());
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return Status::OK();
  }

  Status Accum(OpKernelContext* ctx, const Tensor& keys,
               const Tensor& values_or_deltas, const Tensor& exists) {
    size_t len = keys.flat<K>().size();
    auto stream = ctx->eigen_device<GPUDevice>().stream();
    {
      mutex_lock l(mu_);
      try {
        table_->accum((const K*)keys.tensor_data().data(),
                      (const V*)(values_or_deltas.tensor_data().data()),
                      (const bool*)exists.tensor_data().data(), len, stream);
      } catch (std::runtime_error& e) {
        return Status(tensorflow::error::INTERNAL, e.what());
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return Status::OK();
  }

  Status Remove(OpKernelContext* ctx, const Tensor& keys) override {
    size_t len = keys.flat<K>().size();
    K* d_keys;
    auto stream = ctx->eigen_device<GPUDevice>().stream();

    if (len > 0) {
      CUDA_CHECK(cudaMallocAsync((void**)&d_keys, sizeof(K) * len, stream));
      CUDA_CHECK(cudaMemsetAsync((void*)d_keys, 0, sizeof(K) * len, stream));
      CUDA_CHECK(cudaMemcpyAsync((void*)d_keys,
                                 (void*)keys.tensor_data().data(),
                                 sizeof(K) * len, cudaMemcpyDefault, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      {
        mutex_lock l(mu_);
        try {
          table_->remove((const K*)d_keys, len, stream);
        } catch (std::runtime_error& e) {
          return Status(tensorflow::error::INTERNAL, e.what());
        }
      }
      CUDA_CHECK(cudaFreeAsync(d_keys, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    return Status::OK();
  }

  Status Clear(OpKernelContext* ctx) {
    auto stream = ctx->eigen_device<GPUDevice>().stream();
    {
      mutex_lock l(mu_);
      try {
        table_->clear(stream);
      } catch (std::runtime_error& e) {
        return Status(tensorflow::error::INTERNAL, e.what());
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return Status::OK();
  }

  Status ImportValues(OpKernelContext* ctx, const Tensor& keys,
                      const Tensor& values) override {
    size_t len = keys.flat<K>().size();
    K* d_keys;
    V* d_values;
    if (len > 0) {
      auto stream = ctx->eigen_device<GPUDevice>().stream();
      cudaPointerAttributes keys_attr;
      CUDA_CHECK(cudaPointerGetAttributes(&keys_attr,
                                          (void*)keys.tensor_data().data()));
      if (keys_attr.type != cudaMemoryTypeDevice) {
        CUDA_CHECK(cudaMallocManaged((void**)&d_keys, sizeof(K) * len));
        CUDA_CHECK(cudaMemcpy((void*)d_keys, (void*)keys.tensor_data().data(),
                              sizeof(K) * len, cudaMemcpyDefault));
      } else {
        d_keys = (K*)keys.tensor_data().data();
      }
      cudaPointerAttributes values_attr;
      CUDA_CHECK(cudaPointerGetAttributes(&values_attr,
                                          (void*)values.tensor_data().data()));
      if (values_attr.type != cudaMemoryTypeDevice) {
        CUDA_CHECK(cudaMallocManaged((void**)&d_values,
                                     sizeof(V) * runtime_dim_ * len));
        CUDA_CHECK(
            cudaMemcpy((void*)d_values, (void*)values.tensor_data().data(),
                       sizeof(V) * runtime_dim_ * len, cudaMemcpyDefault));
      } else {
        d_values = (V*)values.tensor_data().data();
      }
      {
        mutex_lock l(mu_);
        try {
          table_->clear(stream);
          table_->upsert((const K*)d_keys, (const V*)d_values, len, stream);
          CUDA_CHECK(cudaStreamSynchronize(stream));
        } catch (std::runtime_error& e) {
          return Status(tensorflow::error::INTERNAL, e.what());
        }
      }
      if (keys_attr.type != cudaMemoryTypeDevice) {
        CUDA_CHECK(cudaFree(d_keys));
      }
      if (values_attr.type != cudaMemoryTypeDevice) {
        CUDA_CHECK(cudaFree(d_values));
      }
    }
    return Status::OK();
  }

  Status ExportValues(OpKernelContext* ctx) override {
    size_t len = 0;
    int64 size = 0;

    const size_t offset = 0;

    Tensor* keys;
    Tensor* values;

    size_t* d_dump_counter = nullptr;
    auto stream = ctx->eigen_device<GPUDevice>().stream();

    {
      tf_shared_lock l(mu_);
      len = table_->get_capacity();
      size = (int64)table_->get_size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaMallocAsync(&d_dump_counter, sizeof(size_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_dump_counter, 0, sizeof(size_t), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    AllocatorAttributes attr;
    // attr.set_gpu_compatible(true);
    // attr.set_nic_compatible(true);
    attr.set_on_host(false);

    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({(size)}), &keys, attr));
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "values", TensorShape({size, (int64)runtime_dim_}), &values, attr));
    if (size) {
      tf_shared_lock l(mu_);
      try {
        table_->dump((K*)keys->flat<K>().data(),
                     (V*)(values->matrix<V>().data()), offset, len,
                     d_dump_counter, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      } catch (std::runtime_error& e) {
        return Status(tensorflow::error::INTERNAL, e.what());
      }
    }
    CUDA_CHECK(cudaFreeAsync(d_dump_counter, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return Status::OK();
  }

  Status ExportValuesWithScores(OpKernelContext* ctx) {
    size_t len = 0;
    int64 size = 0;

    const size_t offset = 0;

    Tensor* keys;
    Tensor* values;
    Tensor* scores;

    size_t* d_dump_counter = nullptr;
    auto stream = ctx->eigen_device<GPUDevice>().stream();

    {
      tf_shared_lock l(mu_);
      len = table_->get_capacity();
      size = (int64)table_->get_size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaMallocAsync(&d_dump_counter, sizeof(size_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_dump_counter, 0, sizeof(size_t), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    AllocatorAttributes attr;
    // attr.set_gpu_compatible(true);
    // attr.set_nic_compatible(true);
    attr.set_on_host(false);

    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({(size)}), &keys, attr));
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "values", TensorShape({size, (int64)runtime_dim_}), &values, attr));
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("scores", TensorShape({(size)}), &scores, attr));
    if (size) {
      tf_shared_lock l(mu_);
      try {
        table_->dump_with_scores((K*)keys->flat<K>().data(),
                                 (V*)(values->matrix<V>().data()),
                                 (uint64_t*)(scores->flat<V>().data()), offset,
                                 len, d_dump_counter, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      } catch (std::runtime_error& e) {
        return Status(tensorflow::error::INTERNAL, e.what());
      }
    }
    CUDA_CHECK(cudaFreeAsync(d_dump_counter, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return Status::OK();
  }

  Status ExportKeysAndScores(OpKernelContext* ctx, size_t split_size) {
    tf_shared_lock l(mu_);
    // size_t span_len = 0;
    int64 size = 0;

    // const size_t offset = 0;

    Tensor* keys = nullptr;
    Tensor* scores = nullptr;

    auto stream = ctx->eigen_device<GPUDevice>().stream();

    AllocatorAttributes attr;
    attr.set_on_host(false);

    {
      size = (int64)table_->get_size(stream);

      TF_RETURN_IF_ERROR(
          ctx->allocate_output("keys", TensorShape({(size)}), &keys, attr));
      TF_RETURN_IF_ERROR(
          ctx->allocate_output("scores", TensorShape({(size)}), &scores, attr));

      if (size) {
        try {
          table_->dump_keys_and_scores((K*)keys->flat<K>().data(),
                                       (int64*)(scores->flat<int64>().data()),
                                       static_cast<size_t>(size), split_size,
                                       stream);
        } catch (std::runtime_error& e) {
          return Status(tensorflow::error::INTERNAL, e.what());
        }
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return Status::OK();
  }

  Status ExportValuesToFile(OpKernelContext* ctx, const string filepath,
                            const size_t buffer_size, bool append_to_file) {
    auto stream = ctx->eigen_device<GPUDevice>().stream();
    FileSystem* fs;
    const auto env = ctx->env();
    TF_RETURN_IF_ERROR(env->GetFileSystemForFile(filepath, &fs));

    {
      tf_shared_lock l(mu_);
      try {
        table_->dump_to_file(fs, filepath, runtime_dim_, stream, buffer_size,
                             append_to_file);
      } catch (std::runtime_error& e) {
        return Status(tensorflow::error::INTERNAL, e.what());
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return Status::OK();
  }

  Status ImportValuesFromFile(OpKernelContext* ctx, const string& dirpath,
                              const std::string& file_name,
                              const size_t buffer_size, bool load_entire_dir) {
    auto stream = ctx->eigen_device<GPUDevice>().stream();
    FileSystem* fs;
    const auto env = ctx->env();
    TF_RETURN_WITH_CONTEXT_IF_ERROR(env->GetFileSystemForFile(dirpath, &fs),
                                    "Please make sure you have already "
                                    "imported tensorflow_io before using "
                                    "TFRA file system operation.");
    const size_t value_dim = static_cast<size_t>(value_shape_.dim_size(0));

    std::vector<std::string> all_filepath;
    std::string filepath = io::JoinPath(dirpath, file_name);

    if (load_entire_dir) {
      string separator = "_mht_";
      int separator_pos = file_name.rfind(separator);
      string file_pattern =
          io::JoinPath(dirpath,
                       file_name.substr(0, separator_pos + separator.size())) +
          "*";
      TF_RETURN_IF_ERROR(fs->GetMatchingPaths(file_pattern, &all_filepath));
      // delete -keys/-values postfix
      for (auto it = all_filepath.begin(); it != all_filepath.end(); ++it) {
        int kv_separator_pos = it->rfind("-");
        *it = it->substr(0, kv_separator_pos);
      }
      // remove duplicate elements
      sort(all_filepath.begin(), all_filepath.end());
      all_filepath.erase(unique(all_filepath.begin(), all_filepath.end()),
                         all_filepath.end());
    }

    {
      mutex_lock l(mu_);
      try {
        table_->clear(stream);
        if (load_entire_dir) {
          for (const auto& path : all_filepath) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            table_->load_from_file(fs, path, runtime_dim_, stream, buffer_size);
          }
        } else {
          CUDA_CHECK(cudaStreamSynchronize(stream));
          table_->load_from_file(fs, filepath, runtime_dim_, stream,
                                 buffer_size);
        }
      } catch (std::runtime_error& e) {
        return Status(tensorflow::error::INTERNAL, e.what());
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return Status::OK();
  }

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }
  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }
  TensorShape key_shape() const final { return TensorShape(); }
  TensorShape value_shape() const override { return value_shape_; }

 private:
  TensorShape value_shape_;
  size_t runtime_dim_;
  mutable mutex mu_;
  gpu::TableWrapper<K, V>* table_ = nullptr GUARDED_BY(mu_);
};

}  // namespace lookup

// Table lookup op. Perform the lookup operation on the given table.
template <class K, class V>
class HashTableFindGpuOp : public OpKernel {
 public:
  explicit HashTableFindGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);
    lookup::HkvHashTableOfTensorsGpu<K, V>* table_hkv =
        (lookup::HkvHashTableOfTensorsGpu<K, V>*)table;

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

    OP_REQUIRES_OK(ctx, table_hkv->Find(ctx, keys, out, default_values));
  }
};

// REGISTER_KERNEL_BUILDER(
//     Name(PREFIX_OP_NAME(HkvHashTableFind)).Device(DEVICE_GPU),
//     HashTableFindGpuOp);

// Table lookup op. Perform the lookup operation on the given table.

template <class K, class V>
class HashTableFindWithExistsGpuOp : public OpKernel {
 public:
  explicit HashTableFindWithExistsGpuOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);

    lookup::HkvHashTableOfTensorsGpu<K, V>* table_hkv =
        (lookup::HkvHashTableOfTensorsGpu<K, V>*)table;

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

    OP_REQUIRES_OK(ctx, table_hkv->FindWithExists(ctx, keys, values,
                                                  default_values, exists));
  }
};

// Table insert op.
class HashTableInsertGpuOp : public OpKernel {
 public:
  explicit HashTableInsertGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

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
    Name(PREFIX_OP_NAME(HkvHashTableInsert)).Device(DEVICE_GPU),
    HashTableInsertGpuOp);

// Table accum op.
template <class K, class V>
class HashTableAccumGpuOp : public OpKernel {
 public:
  explicit HashTableAccumGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);
    lookup::HkvHashTableOfTensorsGpu<K, V>* table_hkv =
        (lookup::HkvHashTableOfTensorsGpu<K, V>*)table;

    DataType expected_input_0 = DT_RESOURCE;
    DataTypeVector expected_inputs = {expected_input_0, table->key_dtype(),
                                      table->value_dtype(),
                                      DataTypeToEnum<bool>::v()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor& keys = ctx->input(1);
    const Tensor& values_or_deltas = ctx->input(2);
    const Tensor& exists = ctx->input(3);
    OP_REQUIRES_OK(
        ctx, table->CheckKeyAndValueTensorsForInsert(keys, values_or_deltas));
    OP_REQUIRES_OK(ctx, table_hkv->Accum(ctx, keys, values_or_deltas, exists));
  }
};

// Table remove op.
// template<class K, class V>
class HashTableRemoveGpuOp : public OpKernel {
 public:
  explicit HashTableRemoveGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

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
    Name(PREFIX_OP_NAME(HkvHashTableRemove)).Device(DEVICE_GPU),
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
    lookup::HkvHashTableOfTensorsGpu<K, V>* table_hkv =
        (lookup::HkvHashTableOfTensorsGpu<K, V>*)table;
    OP_REQUIRES_OK(ctx, table_hkv->Clear(ctx));
  }
};

// Op that returns the size of the given table.
template <class K, class V>
class HashTableSizeGpuOp : public OpKernel {
 public:
  explicit HashTableSizeGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);
    lookup::HkvHashTableOfTensorsGpu<K, V>* table_hkv =
        (lookup::HkvHashTableOfTensorsGpu<K, V>*)table;

    Tensor* out;
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_on_host(false);

    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("size", TensorShape({}), &out, attr));
    int64* p_size = (int64*)out->flat<int64>().data();
    table_hkv->size_i64(ctx, p_size);
  }
};

// REGISTER_KERNEL_BUILDER(
//     Name(PREFIX_OP_NAME(HkvHashTableSize)).Device(DEVICE_GPU),
//     HashTableSizeGpuOp);

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
    Name(PREFIX_OP_NAME(HkvHashTableExport)).Device(DEVICE_GPU),
    HashTableExportGpuOp);

// Op that export all keys and values to file.
template <class K, class V>
class HashTableExportWithScoresGpuOp : public OpKernel {
 public:
  explicit HashTableExportWithScoresGpuOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);
    lookup::HkvHashTableOfTensorsGpu<K, V>* table_hkv =
        (lookup::HkvHashTableOfTensorsGpu<K, V>*)table;
    OP_REQUIRES_OK(ctx, table_hkv->ExportValuesWithScores(ctx));
  }
};

template <class K, class V>
class HashTableExportKeysAndScoresGpuOp : public OpKernel {
 public:
  explicit HashTableExportKeysAndScoresGpuOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    ctx->GetAttr("split_size", &split_size_i64_);
  }

  void Compute(OpKernelContext* ctx) override {
    lookup::LookupInterface* table;
    OP_REQUIRES_OK(ctx, GetLookupTable("table_handle", ctx, &table));
    core::ScopedUnref unref_me(table);
    lookup::HkvHashTableOfTensorsGpu<K, V>* table_hkv =
        (lookup::HkvHashTableOfTensorsGpu<K, V>*)table;
    OP_REQUIRES_OK(ctx, table_hkv->ExportKeysAndScores(
                            ctx, static_cast<size_t>(split_size_i64_)));
  }

 private:
  int64 split_size_i64_;
};

// Clear the table and insert data.
class HashTableImportGpuOp : public OpKernel {
 public:
  explicit HashTableImportGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

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
    Name(PREFIX_OP_NAME(HkvHashTableImport)).Device(DEVICE_GPU),
    HashTableImportGpuOp);

// Op that export all keys and values to FileSystem.
template <class K, class V>
class HashTableSaveToFileSystemGpuOp : public OpKernel {
 public:
  explicit HashTableSaveToFileSystemGpuOp(OpKernelConstruction* ctx)
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

    lookup::HkvHashTableOfTensorsGpu<K, V>* table_hkv =
        (lookup::HkvHashTableOfTensorsGpu<K, V>*)table;
    LOG(INFO) << "c++ dirpath: " << dirpath << " filename: " << file_name;
    std::string filepath = io::JoinPath(dirpath, file_name);

    // OP_REQUIRES_OK(
    //     ctx, fs->RecursivelyCreateDir(std::string(fs->Dirname(filepath))));

    OP_REQUIRES_OK(ctx, table_hkv->ExportValuesToFile(
                            ctx, filepath, buffer_size_, append_to_file_));
  }

 private:
  string dirpath_env_;
  bool append_to_file_;
  size_t buffer_size_;
};

// Clear the table and insert data from FileSystem.
template <class K, class V>
class HashTableLoadFromFileSystemGpuOp : public OpKernel {
 public:
  explicit HashTableLoadFromFileSystemGpuOp(OpKernelConstruction* ctx)
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

    LOG(INFO) << "c++ dirpath :" << dirpath << " filename: " << file_name;

    lookup::HkvHashTableOfTensorsGpu<K, V>* table_hkv =
        (lookup::HkvHashTableOfTensorsGpu<K, V>*)table;
    OP_REQUIRES_OK(
        ctx, table_hkv->ImportValuesFromFile(ctx, dirpath, file_name,
                                             buffer_size_, load_entire_dir_));
  }

 private:
  string dirpath_env_;
  bool load_entire_dir_;
  size_t buffer_size_;
};

// Register the HkvHashTableOfTensors op.
#define REGISTER_KERNEL(key_dtype, value_dtype)                                \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(PREFIX_OP_NAME(HkvHashTableOfTensors))                              \
          .Device(DEVICE_GPU)                                                  \
          .TypeConstraint<key_dtype>("key_dtype")                              \
          .TypeConstraint<value_dtype>("value_dtype"),                         \
      HashTableGpuOp<lookup::HkvHashTableOfTensorsGpu<key_dtype, value_dtype>, \
                     key_dtype, value_dtype>);                                 \
  REGISTER_KERNEL_BUILDER(Name(PREFIX_OP_NAME(HkvHashTableClear))              \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<key_dtype>("key_dtype")          \
                              .TypeConstraint<value_dtype>("value_dtype"),     \
                          HashTableClearGpuOp<key_dtype, value_dtype>);        \
  REGISTER_KERNEL_BUILDER(Name(PREFIX_OP_NAME(HkvHashTableSize))               \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<key_dtype>("key_dtype")          \
                              .TypeConstraint<value_dtype>("value_dtype"),     \
                          HashTableSizeGpuOp<key_dtype, value_dtype>);         \
  REGISTER_KERNEL_BUILDER(Name(PREFIX_OP_NAME(HkvHashTableAccum))              \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<key_dtype>("key_dtype")          \
                              .TypeConstraint<value_dtype>("value_dtype"),     \
                          HashTableAccumGpuOp<key_dtype, value_dtype>);        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(PREFIX_OP_NAME(HkvHashTableExportWithScores))                       \
          .Device(DEVICE_GPU)                                                  \
          .TypeConstraint<key_dtype>("key_dtype")                              \
          .TypeConstraint<value_dtype>("value_dtype"),                         \
      HashTableExportWithScoresGpuOp<key_dtype, value_dtype>);                 \
  REGISTER_KERNEL_BUILDER(Name(PREFIX_OP_NAME(HkvHashTableFind))               \
                              .Device(DEVICE_GPU)                              \
                              .TypeConstraint<key_dtype>("Tin")                \
                              .TypeConstraint<value_dtype>("Tout"),            \
                          HashTableFindGpuOp<key_dtype, value_dtype>);         \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(PREFIX_OP_NAME(HkvHashTableFindWithExists))                         \
          .Device(DEVICE_GPU)                                                  \
          .TypeConstraint<key_dtype>("Tin")                                    \
          .TypeConstraint<value_dtype>("Tout"),                                \
      HashTableFindWithExistsGpuOp<key_dtype, value_dtype>);                   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(PREFIX_OP_NAME(HkvHashTableSaveToFileSystem))                       \
          .Device(DEVICE_GPU)                                                  \
          .TypeConstraint<key_dtype>("key_dtype")                              \
          .TypeConstraint<value_dtype>("value_dtype"),                         \
      HashTableSaveToFileSystemGpuOp<key_dtype, value_dtype>);                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(PREFIX_OP_NAME(HkvHashTableLoadFromFileSystem))                     \
          .Device(DEVICE_GPU)                                                  \
          .TypeConstraint<key_dtype>("key_dtype")                              \
          .TypeConstraint<value_dtype>("value_dtype"),                         \
      HashTableLoadFromFileSystemGpuOp<key_dtype, value_dtype>);

REGISTER_KERNEL(int64, float);
REGISTER_KERNEL(int64, int8);
REGISTER_KERNEL(int64, int32);
REGISTER_KERNEL(int64, int64);
REGISTER_KERNEL(int64, Eigen::half);

#undef REGISTER_KERNEL

#define SINGLE_ATTR_REGISTER_KERNEL(key_dtype, value_type)  \
  REGISTER_KERNEL_BUILDER(                                  \
      Name(PREFIX_OP_NAME(HkvHashTableExportKeysAndScores)) \
          .Device(DEVICE_GPU)                               \
          .TypeConstraint<key_dtype>("Tkeys"),              \
      HashTableExportKeysAndScoresGpuOp<key_dtype, value_type>);

SINGLE_ATTR_REGISTER_KERNEL(int64, float);

#undef SINGLE_ATTR_REGISTER_KERNEL

}  // namespace recommenders_addons
}  // namespace tensorflow
#endif
