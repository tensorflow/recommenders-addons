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
namespace lookup {

using tensorflow::OpKernelContext;
using tensorflow::lookup::LookupInterface;

template <class K, class V>
class CuckooHashTableOfTensorsGpu final : public LookupInterface {
 public:
  CuckooHashTableOfTensorsGpu(OpKernelContext* ctx, OpKernel* kernel)
      : last_hint_size_(0) {
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
      CUDA_CHECK(cudaStreamDestroy(_stream));
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

    cudaStream_t _stream;

    if (len > 0) {
      size_t default_value_num =
          is_full_default ? default_value.shape().dim_size(0) : 1;
      CUDA_CHECK(cudaStreamCreate(&_stream));
      {
        tf_shared_lock l(mu_);
        table_->get((const K*)d_keys.tensor_data().data(),
                    (gpu::ValueArrayBase<V>*)value->tensor_data().data(),
                    (bool*)exists->tensor_data().data(), len,
                    (gpu::ValueArrayBase<V>*)default_value.tensor_data().data(),
                    _stream, is_full_default);
        CUDA_CHECK(cudaStreamSynchronize(_stream));
      }
      CUDA_CHECK(cudaStreamDestroy(_stream));
    }
    return Status::OK();
  }

  void RehashIfNeeded(cudaStream_t stream, const size_t num_keys = 0) {
    last_hint_size_ =
        table_->rehash_if_needed(min_size_, stream, num_keys, last_hint_size_);
    max_size_ = table_->get_capacity();
  }

  Status Insert(OpKernelContext* ctx, const Tensor& keys,
                const Tensor& values) override {
    size_t len = keys.flat<K>().size();
    cudaStream_t _stream;
    CUDA_CHECK(cudaStreamCreate(&_stream));
    {
      mutex_lock l(mu_);
      RehashIfNeeded(_stream, len);
      CUDA_CHECK(cudaDeviceSynchronize());
      table_->upsert((const K*)keys.tensor_data().data(),
                     (const gpu::ValueArrayBase<V>*)values.tensor_data().data(),
                     len, _stream);
      CUDA_CHECK(cudaStreamSynchronize(_stream));
    };
    CUDA_CHECK(cudaStreamDestroy(_stream));

    return Status::OK();
  }

  Status Accum(OpKernelContext* ctx, const Tensor& keys,
               const Tensor& values_or_deltas, const Tensor& exists) {
    size_t len = keys.flat<K>().size();
    cudaStream_t _stream;
    CUDA_CHECK(cudaStreamCreate(&_stream));
    {
      mutex_lock l(mu_);
      RehashIfNeeded(_stream, len);
      CUDA_CHECK(cudaDeviceSynchronize());
      table_->accum(
          (const K*)keys.tensor_data().data(),
          (const gpu::ValueArrayBase<V>*)values_or_deltas.tensor_data().data(),
          (const bool*)exists.tensor_data().data(), len, _stream);
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
      {
        mutex_lock l(mu_);
        table_->remove((const K*)d_keys, len, _stream);
        RehashIfNeeded(_stream);
        CUDA_CHECK(cudaStreamSynchronize(_stream));
      }
      CUDA_CHECK(cudaStreamDestroy(_stream));
      if (keys_attr.type != cudaMemoryTypeDevice) {
        CUDA_CHECK(cudaFree(d_keys));
      }
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
        d_values = (gpu::ValueArrayBase<V>*)values.tensor_data().data();
      }
      {
        mutex_lock l(mu_);
        table_->clear(_stream);
        RehashIfNeeded(_stream, len);
        CUDA_CHECK(cudaDeviceSynchronize());
        table_->upsert((const K*)d_keys,
                       (const gpu::ValueArrayBase<V>*)d_values, len, _stream);
        CUDA_CHECK(cudaStreamSynchronize(_stream));
      }
      CUDA_CHECK(cudaStreamDestroy(_stream));
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

// For propagating errors when calling a function.
#define TF_RETURN_IF_ERROR_WITH_CLEANUP_CUDA(CLEANUP_CODE, ...) \
  do {                                                          \
    ::tensorflow::Status _status = (__VA_ARGS__);               \
    if (TF_PREDICT_FALSE(!_status.ok())) {                      \
      {CLEANUP_CODE};                                           \
      return _status;                                           \
    }                                                           \
  } while (0)

  Status SaveToFileSystemImpl(FileSystem* fs, const size_t value_dim,
                              const string& filepath, const size_t buffer_size,
                              bool append_to_file, cudaStream_t stream) {
    std::unique_ptr<WritableFile> key_writer;
    std::unique_ptr<WritableFile> value_writer;
    const string key_filepath(filepath + "-keys");
    const string value_filepath(filepath + "-values");
    string key_tmpfilepath(filepath + "-keys.tmp");
    string value_tmpfilepath(filepath + "-values.tmp");
    bool has_atomic_move = false;
    auto has_atomic_move_ret = fs->HasAtomicMove(filepath, &has_atomic_move);
    bool need_tmp_file =
        (has_atomic_move == false) || (has_atomic_move_ret != Status::OK());
    if (!need_tmp_file) {
      key_tmpfilepath = key_filepath;
      value_tmpfilepath = value_filepath;
    }
    TF_RETURN_IF_ERROR(
        fs->RecursivelyCreateDir(std::string(fs->Dirname(filepath))));
    if (append_to_file) {
      TF_RETURN_IF_ERROR(fs->NewAppendableFile(key_tmpfilepath, &key_writer));
      TF_RETURN_IF_ERROR(
          fs->NewAppendableFile(value_tmpfilepath, &value_writer));
    } else {
      TF_RETURN_IF_ERROR(fs->NewWritableFile(key_tmpfilepath, &key_writer));
      TF_RETURN_IF_ERROR(fs->NewWritableFile(value_tmpfilepath, &value_writer));
    }

    size_t key_offset = 0;
    size_t value_offset = 0;
    const size_t value_len = sizeof(V) * value_dim;
    const size_t key_buffer_byte_size = buffer_size * sizeof(K);
    const size_t value_buffer_byte_size = buffer_size * value_len;
    std::vector<char> key_buffer_vector(key_buffer_byte_size);
    char* key_buffer = key_buffer_vector.data();
    std::vector<char> value_buffer_vector(value_buffer_byte_size);
    char* value_buffer = value_buffer_vector.data();

    K* d_keys = nullptr;
    V* d_values = nullptr;
    size_t* d_dump_counter;
    size_t dump_counter;
    size_t search_offset = 0;
    size_t table_capacity = table_->get_capacity();

    CUDA_CHECK(cudaMallocAsync(&d_keys, key_buffer_byte_size, stream));
    CUDA_CHECK(cudaMallocAsync(&d_values, value_buffer_byte_size, stream));
    CUDA_CHECK(cudaMallocAsync(&d_dump_counter, sizeof(size_t), stream));
#define CLEANUP_CUDA_CODE                      \
  CUDA_CHECK(cudaFreeAsync(d_keys, stream));   \
  CUDA_CHECK(cudaFreeAsync(d_values, stream)); \
  CUDA_CHECK(cudaFreeAsync(d_dump_counter, stream));

    size_t search_length = 0;
    size_t total_saved = 0;
    while (search_offset < table_capacity) {
      if (search_offset + buffer_size >= table_capacity) {
        search_length = table_capacity - search_offset;
      } else {
        search_length = buffer_size;
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      table_->dump(d_keys, (gpu::ValueType<V>*)d_values, search_offset,
                   search_length, d_dump_counter, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      CUDA_CHECK(cudaMemcpyAsync(&dump_counter, d_dump_counter, sizeof(size_t),
                                 cudaMemcpyDeviceToHost, stream));

      if (dump_counter > 0) {
        key_offset = dump_counter * sizeof(K);
        value_offset = dump_counter * value_len;
        CUDA_CHECK(cudaMemcpyAsync(key_buffer, d_keys, key_offset,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(value_buffer, d_values, value_offset,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        TF_RETURN_IF_ERROR_WITH_CLEANUP_CUDA(
            CLEANUP_CUDA_CODE,
            key_writer->Append(StringPiece(key_buffer, key_offset)));
        TF_RETURN_IF_ERROR_WITH_CLEANUP_CUDA(
            CLEANUP_CUDA_CODE,
            value_writer->Append(StringPiece(value_buffer, value_offset)));
      }
      search_offset += search_length;
      total_saved += dump_counter;
    }

    CLEANUP_CUDA_CODE
#undef CLEANUP_CUDA_CODE

    TF_RETURN_IF_ERROR(key_writer->Flush());
    TF_RETURN_IF_ERROR(value_writer->Flush());
    // "munmap_chunk(): invalid pointer" when call TF IO S3 File System Sync()
    // function, unknown reasons.
    // TODO: Fix it.
    // TF_RETURN_IF_ERROR(key_writer->Sync());
    // TF_RETURN_IF_ERROR(value_writer->Sync());

    LOG(INFO) << "Finish saving " << total_saved << " keys and values to "
              << key_filepath << " and " << value_filepath << " in total.";

    if (need_tmp_file) {
      TF_RETURN_IF_ERROR(fs->FileExists(key_tmpfilepath));
      TF_RETURN_IF_ERROR(fs->RenameFile(key_tmpfilepath, key_filepath));
      TF_RETURN_IF_ERROR(fs->FileExists(value_tmpfilepath));
      TF_RETURN_IF_ERROR(fs->RenameFile(value_tmpfilepath, value_filepath));
    }

    return Status::OK();
  }

  Status SaveToFileSystem(OpKernelContext* ctx, const string& dirpath,
                          const string& file_name, const size_t buffer_size,
                          bool append_to_file) {
    string filepath = io::JoinPath(dirpath, file_name);
    FileSystem* fs;
    const auto env = ctx->env();
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        env->GetFileSystemForFile(filepath, &fs),
        "Please make sure you have already imported tensorflow_io before using "
        "TFRA file system operation.");
    const size_t value_dim = static_cast<size_t>(value_shape_.dim_size(0));
    cudaStream_t _stream;
    CUDA_CHECK(cudaStreamCreate(&_stream));
    auto statu = SaveToFileSystemImpl(fs, value_dim, filepath, buffer_size,
                                      append_to_file, _stream);
    CUDA_CHECK(cudaStreamDestroy(_stream));
    return statu;
  }

  Status LoadFromFileSystemImpl(FileSystem* fs, const size_t value_dim,
                                const string& filepath,
                                const size_t buffer_size, cudaStream_t stream) {
    const string key_filepath = filepath + "-keys";
    TF_RETURN_IF_ERROR(fs->FileExists(key_filepath));
    std::unique_ptr<RandomAccessFile> key_file;
    TF_RETURN_IF_ERROR(fs->NewRandomAccessFile(key_filepath, &key_file));
    std::unique_ptr<io::RandomAccessInputStream> key_input_stream(
        new io::RandomAccessInputStream(key_file.get()));
    const size_t key_buffer_byte_size = buffer_size * sizeof(K);
    io::BufferedInputStream key_reader(key_input_stream.get(),
                                       key_buffer_byte_size * 2);

    const string value_filepath = filepath + "-values";
    TF_RETURN_IF_ERROR(fs->FileExists(value_filepath));
    std::unique_ptr<RandomAccessFile> value_file;
    TF_RETURN_IF_ERROR(fs->NewRandomAccessFile(value_filepath, &value_file));
    std::unique_ptr<io::RandomAccessInputStream> value_input_stream(
        new io::RandomAccessInputStream(value_file.get()));
    const size_t value_len = sizeof(V) * value_dim;
    const size_t value_buffer_byte_size = buffer_size * value_len;
    io::BufferedInputStream value_reader(value_input_stream.get(),
                                         value_buffer_byte_size * 2);

    uint64 key_file_size = 0;
    TF_RETURN_IF_ERROR(fs->GetFileSize(key_filepath, &key_file_size));
    const size_t key_size = key_file_size / sizeof(K);

    uint64 value_file_size = 0;
    TF_RETURN_IF_ERROR(fs->GetFileSize(value_filepath, &value_file_size));
    const size_t value_size = value_file_size / value_len;

    if (key_size != value_size) {
      return errors::Unavailable(
          "the keys number in file " + key_filepath +
          " is not equal to the value vectors number in file " +
          value_filepath + ".");
    }

    // Rehash table
    RehashIfNeeded(stream, key_size);

    K* d_keys = nullptr;
    V* d_values = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_keys, key_buffer_byte_size, stream));
    CUDA_CHECK(cudaMallocAsync(&d_values, value_buffer_byte_size, stream));
#define CLEANUP_CUDA_CODE                    \
  CUDA_CHECK(cudaFreeAsync(d_keys, stream)); \
  CUDA_CHECK(cudaFreeAsync(d_values, stream));

    tstring key_buffer;
    key_buffer.resize(key_buffer_byte_size);
    tstring value_buffer;
    value_buffer.resize(value_buffer_byte_size);

    size_t key_file_offset = 0;
    int64_t remainder = key_file_size - key_file_offset;
    size_t nkeys = 0;
    size_t key_read_byte = 0;
    size_t value_read_byte = 0;
    while (remainder > 0) {
      if (remainder > static_cast<int64_t>(key_buffer_byte_size)) {
        key_read_byte = key_buffer_byte_size;
        nkeys = buffer_size;
        value_read_byte = value_buffer_byte_size;
      } else {
        key_read_byte = remainder;
        nkeys = key_read_byte / sizeof(K);
        value_read_byte = nkeys * value_len;
      }
      TF_RETURN_IF_ERROR_WITH_CLEANUP_CUDA(
          CLEANUP_CUDA_CODE, key_reader.ReadNBytes(key_read_byte, &key_buffer));
      TF_RETURN_IF_ERROR_WITH_CLEANUP_CUDA(
          CLEANUP_CUDA_CODE,
          value_reader.ReadNBytes(value_read_byte, &value_buffer));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaMemcpyAsync(d_keys, key_buffer.data(), key_read_byte,
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_values, value_buffer.data(), value_read_byte,
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      table_->upsert(d_keys, (gpu::ValueType<V>*)d_values, nkeys, stream);
      key_file_offset += key_read_byte;
      remainder = key_file_size - key_file_offset;
    }

    CLEANUP_CUDA_CODE
#undef CLEANUP_CUDA_CODE
    LOG(INFO) << "Finish loading " << key_size << " keys and values from "
              << key_filepath << " and " << value_filepath << " in total.";

    return Status::OK();
  }

  Status LoadFromFileSystem(OpKernelContext* ctx, const string& dirpath,
                            const string& file_name, const size_t buffer_size,
                            bool load_entire_dir) {
    FileSystem* fs;
    const auto env = ctx->env();
    TF_RETURN_WITH_CONTEXT_IF_ERROR(env->GetFileSystemForFile(dirpath, &fs),
                                    "Please make sure you have already "
                                    "imported tensorflow_io before using "
                                    "TFRA file system operation.");
    const size_t value_dim = static_cast<size_t>(value_shape_.dim_size(0));
    auto statu = Status::OK();
    if (load_entire_dir) {
      string separator = "_mht_";
      int separator_pos = file_name.rfind(separator);
      string file_pattern =
          io::JoinPath(dirpath,
                       file_name.substr(0, separator_pos + separator.size())) +
          "*";
      std::vector<string> all_filepath;
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
      for (auto fp : all_filepath) {
        cudaStream_t _stream;
        CUDA_CHECK(cudaStreamCreate(&_stream));
        statu = LoadFromFileSystemImpl(fs, value_dim, fp, buffer_size, _stream);
        CUDA_CHECK(cudaStreamDestroy(_stream));
        if (statu != Status::OK()) {
          return statu;
        }
      }
    } else {
      string filepath = io::JoinPath(dirpath, file_name);
      cudaStream_t _stream;
      CUDA_CHECK(cudaStreamCreate(&_stream));
      statu =
          LoadFromFileSystemImpl(fs, value_dim, filepath, buffer_size, _stream);
      CUDA_CHECK(cudaStreamDestroy(_stream));
    }
    return statu;
  }

#undef TF_RETURN_IF_ERROR_WITH_CLEANUP_CUDA

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }
  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }
  TensorShape key_shape() const final { return TensorShape(); }
  TensorShape value_shape() const override { return value_shape_; }

 private:
  TensorShape value_shape_;
  size_t max_size_;
  size_t min_size_;
  size_t last_hint_size_;
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
    Name(PREFIX_OP_NAME(CuckooHashTableFind)).Device(DEVICE_GPU),
    HashTableFindGpuOp);

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

    lookup::CuckooHashTableOfTensorsGpu<K, V>* table_cuckoo =
        (lookup::CuckooHashTableOfTensorsGpu<K, V>*)table;

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

    OP_REQUIRES_OK(ctx, table_cuckoo->FindWithExists(ctx, keys, values,
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
    Name(PREFIX_OP_NAME(CuckooHashTableInsert)).Device(DEVICE_GPU),
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
    lookup::CuckooHashTableOfTensorsGpu<K, V>* table_cuckoo =
        (lookup::CuckooHashTableOfTensorsGpu<K, V>*)table;

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
    OP_REQUIRES_OK(ctx,
                   table_cuckoo->Accum(ctx, keys, values_or_deltas, exists));
  }
};

// Table remove op.
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

    lookup::CuckooHashTableOfTensorsGpu<K, V>* table_cuckoo =
        (lookup::CuckooHashTableOfTensorsGpu<K, V>*)table;
    OP_REQUIRES_OK(
        ctx, table_cuckoo->SaveToFileSystem(ctx, dirpath, file_name,
                                            buffer_size_, append_to_file_));
  }

 private:
  string dirpath_env_;
  bool append_to_file_;
  size_t buffer_size_;
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
    Name(PREFIX_OP_NAME(CuckooHashTableImport)).Device(DEVICE_GPU),
    HashTableImportGpuOp);

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

    lookup::CuckooHashTableOfTensorsGpu<K, V>* table_cuckoo =
        (lookup::CuckooHashTableOfTensorsGpu<K, V>*)table;
    OP_REQUIRES_OK(
        ctx, table_cuckoo->LoadFromFileSystem(ctx, dirpath, file_name,
                                              buffer_size_, load_entire_dir_));
  }

 private:
  string dirpath_env_;
  bool load_entire_dir_;
  size_t buffer_size_;
};

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
                          HashTableClearGpuOp<key_dtype, value_dtype>)     \
  REGISTER_KERNEL_BUILDER(Name(PREFIX_OP_NAME(CuckooHashTableAccum))       \
                              .Device(DEVICE_GPU)                          \
                              .TypeConstraint<key_dtype>("key_dtype")      \
                              .TypeConstraint<value_dtype>("value_dtype"), \
                          HashTableAccumGpuOp<key_dtype, value_dtype>)     \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name(PREFIX_OP_NAME(CuckooHashTableFindWithExists))                  \
          .Device(DEVICE_GPU)                                              \
          .TypeConstraint<key_dtype>("Tin")                                \
          .TypeConstraint<value_dtype>("Tout"),                            \
      HashTableFindWithExistsGpuOp<key_dtype, value_dtype>)                \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name(PREFIX_OP_NAME(CuckooHashTableSaveToFileSystem))                \
          .Device(DEVICE_GPU)                                              \
          .TypeConstraint<key_dtype>("key_dtype")                          \
          .TypeConstraint<value_dtype>("value_dtype"),                     \
      HashTableSaveToFileSystemGpuOp<key_dtype, value_dtype>);             \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name(PREFIX_OP_NAME(CuckooHashTableLoadFromFileSystem))              \
          .Device(DEVICE_GPU)                                              \
          .TypeConstraint<key_dtype>("key_dtype")                          \
          .TypeConstraint<value_dtype>("value_dtype"),                     \
      HashTableLoadFromFileSystemGpuOp<key_dtype, value_dtype>);

REGISTER_KERNEL(int64, float);
REGISTER_KERNEL(int64, Eigen::half);
REGISTER_KERNEL(int64, int64);
REGISTER_KERNEL(int64, int32);
REGISTER_KERNEL(int64, int8);
REGISTER_KERNEL(int32, float);

#undef REGISTER_KERNEL

}  // namespace recommenders_addons
}  // namespace tensorflow
#endif
