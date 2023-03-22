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

#include <string>
#include <type_traits>
#include <utility>

#include "lookup_table_interface.h"
#include "lookup_table_op_registry.h"

#include "tensorflow/core/kernels/lookup_table_op.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/kernels/lookup_impl/lookup_table_op_cpu.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/types.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/utils.h"

namespace tensorflow {
namespace recommenders_addons {
namespace lookup {
typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace lookup_table;

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
    int64 num_keys = key_flat.size();

    auto shard = [this, table, key_flat, &value_flat, &default_flat,
                  &is_full_default](int64 begin, int64 end) {
      for (int64 i = begin; i < end; ++i) {
        table->find(key_flat(i), value_flat, default_flat, value_dim_,
                    is_full_default, i);
      }
    };
    auto& worker_threads = *context->device()->tensorflow_cpu_worker_threads();
    int64 slices = static_cast<int64>(total / worker_threads.num_threads) + 1;
    Shard(worker_threads.num_threads, worker_threads.workers, num_keys, slices,
          shard);
  }

 private:
  const int64 value_dim_;
};

template <typename Device, class K, class V>
struct LaunchTensorsFindWithExists;

template <class K, class V>
struct LaunchTensorsFindWithExists<CPUDevice, K, V> {
  explicit LaunchTensorsFindWithExists(int64 value_dim)
      : value_dim_(value_dim) {}

  void launch(OpKernelContext* context, cpu::TableWrapperBase<K, V>* table,
              const Tensor& key, Tensor* value, const Tensor& default_value,
              Tensor& exists) {
    const auto key_flat = key.flat<K>();
    cpu::Tensor2D<V> value_flat = value->flat_inner_dims<V, 2>();
    cpu::ConstTensor2D<V> default_flat = default_value.flat_inner_dims<V, 2>();
    auto exists_flat = exists.flat<bool>();

    int64 total = value_flat.size();
    int64 default_total = default_flat.size();
    bool is_full_default = (total == default_total);
    int64 num_keys = key_flat.size();

    auto shard = [this, table, key_flat, &value_flat, &default_flat,
                  &exists_flat, &is_full_default](int64 begin, int64 end) {
      for (int64 i = begin; i < end; ++i) {
        table->find(key_flat(i), value_flat, default_flat, exists_flat(i),
                    value_dim_, is_full_default, i);
      }
    };
    auto& worker_threads = *context->device()->tensorflow_cpu_worker_threads();
    int64 slices = static_cast<int64>(total / worker_threads.num_threads) + 1;
    Shard(worker_threads.num_threads, worker_threads.workers, num_keys, slices,
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

template <typename Device, class K, class V>
struct LaunchTensorsAccum;

template <class K, class V>
struct LaunchTensorsAccum<CPUDevice, K, V> {
  explicit LaunchTensorsAccum(int64 value_dim) : value_dim_(value_dim) {}

  void launch(OpKernelContext* context, cpu::TableWrapperBase<K, V>* table,
              const Tensor& keys, const Tensor& values_or_deltas,
              const Tensor& exists) {
    const auto key_flat = keys.flat<K>();
    int64 total = key_flat.size();
    const auto values_or_deltas_flat = values_or_deltas.flat_inner_dims<V, 2>();
    const auto exist_flat = exists.flat<bool>();

    auto shard = [this, &table, key_flat, &values_or_deltas_flat, &exist_flat](
                     int64 begin, int64 end) {
      for (int64 i = begin; i < end; ++i) {
        table->insert_or_accum(key_flat(i), values_or_deltas_flat,
                               exist_flat(i), value_dim_, i);
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

template <class K, class V>
class CuckooHashTable final : public LookupTableInterface {
 public:
  CuckooHashTable(OpKernelContext* ctx, OpKernel* kernel) {
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

  ~CuckooHashTable() { delete table_; }

  size_t size() const override { return table_->size(); }

  Status Find(OpKernelContext* ctx, const Tensor& key, Tensor* value,
              const Tensor& default_value) override {
    int64 value_dim = value_shape_.dim_size(0);

    LaunchTensorsFind<CPUDevice, K, V> launcher(value_dim);
    launcher.launch(ctx, table_, key, value, default_value);

    return Status::OK();
  }

  Status FindWithExists(OpKernelContext* ctx, const Tensor& key, Tensor* value,
                        const Tensor& default_value, Tensor& exists) {
    int64 value_dim = value_shape_.dim_size(0);

    LaunchTensorsFindWithExists<CPUDevice, K, V> launcher(value_dim);
    launcher.launch(ctx, table_, key, value, default_value, exists);

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

  Status DoAccum(bool clear, OpKernelContext* ctx, const Tensor& keys,
                 const Tensor& values_or_deltas, const Tensor& exists) {
    int64 value_dim = value_shape_.dim_size(0);

    if (clear) {
      table_->clear();
    }

    LaunchTensorsAccum<CPUDevice, K, V> launcher(value_dim);
    launcher.launch(ctx, table_, keys, values_or_deltas, exists);

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

  Status Clear(OpKernelContext* ctx) {
    table_->clear();
    return Status::OK();
  }

  Status Accum(OpKernelContext* ctx, const Tensor& keys,
               const Tensor& values_or_deltas, const Tensor& exists) {
    return DoAccum(false, ctx, keys, values_or_deltas, exists);
  }

  Status ImportValues(OpKernelContext* ctx, const Tensor& keys,
                      const Tensor& values) override {
    return DoInsert(true, ctx, keys, values);
  }

  Status ExportValues(OpKernelContext* ctx) override {
    Tensor* keys;
    Tensor* values;
    const auto table_size = table_->size();
    const auto output_key_size = static_cast<int64>(table_size);
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({output_key_size}), &keys));
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "values",
        TensorShape({output_key_size, static_cast<int64>(runtime_dim_)}),
        &values));
    table_->dump((K*)keys->tensor_data().data(),
                 (V*)values->tensor_data().data(), 0, table_size);

    return Status::OK();
  }

  Status SaveToFileSystemImpl(FileSystem* fs, const size_t value_dim,
                              const string& filepath, const size_t buffer_size,
                              bool append_to_file) {
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

    const size_t table_size = table_->size();
    size_t search_offset = 0;
    size_t total_saved = 0;
    while (search_offset < table_size) {
      auto dump_counter = table_->dump((K*)key_buffer, (V*)value_buffer,
                                       search_offset, buffer_size);
      search_offset += dump_counter;
      key_offset += dump_counter * sizeof(K);
      value_offset += dump_counter * value_len;
      TF_RETURN_IF_ERROR(
          key_writer->Append(StringPiece(key_buffer, key_offset)));
      key_buffer = key_buffer_vector.data();
      key_offset = 0;
      TF_RETURN_IF_ERROR(
          value_writer->Append(StringPiece(value_buffer, value_offset)));
      value_buffer = value_buffer_vector.data();
      value_offset = 0;
      total_saved += dump_counter;
    }

    if (key_offset > 0 && value_offset > 0) {
      TF_RETURN_IF_ERROR(
          key_writer->Append(StringPiece(key_buffer, key_offset)));
      TF_RETURN_IF_ERROR(
          value_writer->Append(StringPiece(value_buffer, value_offset)));
    }

    TF_RETURN_IF_ERROR(key_writer->Flush());
    TF_RETURN_IF_ERROR(value_writer->Flush());
    TF_RETURN_IF_ERROR(key_writer->Sync());
    TF_RETURN_IF_ERROR(value_writer->Sync());

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
    return SaveToFileSystemImpl(fs, value_dim, filepath, buffer_size,
                                append_to_file);
  }

  Status LoadFromFileSystemImpl(FileSystem* fs, const size_t value_dim,
                                const string& filepath,
                                const size_t buffer_size) {
    const string key_filepath = filepath + "-keys";
    TF_RETURN_IF_ERROR(fs->FileExists(key_filepath));
    std::unique_ptr<RandomAccessFile> key_file;
    TF_RETURN_IF_ERROR(fs->NewRandomAccessFile(key_filepath, &key_file));
    std::unique_ptr<io::RandomAccessInputStream> key_input_stream(
        new io::RandomAccessInputStream(key_file.get()));
    const size_t key_buffer_byte_size = buffer_size * sizeof(K);
    io::BufferedInputStream key_reader(key_input_stream.get(),
                                       key_buffer_byte_size);

    const string value_filepath = filepath + "-values";
    TF_RETURN_IF_ERROR(fs->FileExists(value_filepath));
    std::unique_ptr<RandomAccessFile> value_file;
    TF_RETURN_IF_ERROR(fs->NewRandomAccessFile(value_filepath, &value_file));
    std::unique_ptr<io::RandomAccessInputStream> value_input_stream(
        new io::RandomAccessInputStream(value_file.get()));
    const size_t value_len = sizeof(V) * value_dim;
    const size_t value_buffer_size = buffer_size * value_len;
    io::BufferedInputStream value_reader(value_input_stream.get(),
                                         value_buffer_size);

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

    tstring key_buffer;
    key_buffer.resize(sizeof(K));
    tstring value_buffer;
    value_buffer.resize(value_len);
    uint64 key_file_offset = 0;

    while (key_file_offset < key_file_size) {
      TF_RETURN_IF_ERROR(key_reader.ReadNBytes(sizeof(K), &key_buffer));
      TF_RETURN_IF_ERROR(value_reader.ReadNBytes(value_len, &value_buffer));
      table_->insert_or_assign((K*)key_buffer.data(), (V*)value_buffer.data(),
                               runtime_dim_);
      key_file_offset += sizeof(K);
    }

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
    if (load_entire_dir) {
      int separator_pos = file_name.rfind("_mht_");
      string file_pattern =
          io::JoinPath(dirpath, file_name.substr(0, separator_pos)) + "*";
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
      for (auto& fp : all_filepath) {
        TF_RETURN_IF_ERROR(
            LoadFromFileSystemImpl(fs, value_dim, fp, buffer_size));
      }
    } else {
      string filepath = io::JoinPath(dirpath, file_name);
      return LoadFromFileSystemImpl(fs, value_dim, filepath, buffer_size);
    }
    return Status::OK();
  }

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }

  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }

  TensorShape key_shape() const final { return TensorShape(); }

  TensorShape value_shape() const override { return value_shape_; }

  int64 MemoryUsed() const override {
    int64 ret = 0;
    ret = (int64)table_->size();
    return sizeof(CuckooHashTable) + ret;
  }

 private:
  TensorShape value_shape_;
  size_t runtime_dim_;
  cpu::TableWrapperBase<K, V>* table_ = nullptr;
  size_t init_size_;
};  // class CuckooHashTable


#define REGISTER_CUCKOO_HASHTABLE(key_dtype, value_dtype)                     \
  REGISTER_LOOKUP_TABLE("CUCKOO_HASHTABLE", "CPU", key_dtype, value_dtype,    \
      CuckooHashTable<key_dtype, value_dtype>);

REGISTER_CUCKOO_HASHTABLE(int32, double);
REGISTER_CUCKOO_HASHTABLE(int32, float);
REGISTER_CUCKOO_HASHTABLE(int32, int32);
REGISTER_CUCKOO_HASHTABLE(int64, double);
REGISTER_CUCKOO_HASHTABLE(int64, float);
REGISTER_CUCKOO_HASHTABLE(int64, int32);
REGISTER_CUCKOO_HASHTABLE(int64, int64);
REGISTER_CUCKOO_HASHTABLE(int64, tstring);
REGISTER_CUCKOO_HASHTABLE(int64, int8);
REGISTER_CUCKOO_HASHTABLE(int64, Eigen::half);
REGISTER_CUCKOO_HASHTABLE(int64, bfloat16);
REGISTER_CUCKOO_HASHTABLE(tstring, bool);
REGISTER_CUCKOO_HASHTABLE(tstring, double);
REGISTER_CUCKOO_HASHTABLE(tstring, float);
REGISTER_CUCKOO_HASHTABLE(tstring, int32);
REGISTER_CUCKOO_HASHTABLE(tstring, int64);
REGISTER_CUCKOO_HASHTABLE(tstring, int8);
REGISTER_CUCKOO_HASHTABLE(tstring, Eigen::half);
REGISTER_CUCKOO_HASHTABLE(tstring, bfloat16);

#undef REGISTER_CUCKOO_HASHTABLE

}  // namespace lookup
}  // namespace recommenders_addons
}  // namespace tensorflow
