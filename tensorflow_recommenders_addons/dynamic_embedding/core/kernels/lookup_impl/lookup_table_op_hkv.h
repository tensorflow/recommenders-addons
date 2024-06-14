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

#ifndef TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_HKV_H_
#define TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_HKV_H_

#include <stddef.h>
#include <stdio.h>
#include <time.h>

#include <limits>
#include <string>
#include <typeindex>
#include <vector>

#include "merlin/allocator.cuh"
#include "merlin/types.cuh"
#include "merlin/utils.cuh"
#include "merlin_hashtable.cuh"
#include "merlin_localfile.hpp"
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
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/utils.h"

namespace tensorflow {
namespace recommenders_addons {
namespace hkv_table {
namespace gpu {

inline Status ReturnInternalErrorStatus(const char* const str) {
#if TF_VERSION_INTEGER >= 2130 /* 2.13.0 */
  return Status(absl::StatusCode::kInternal, str);
#else
  return Status(tensorflow::error::INTERNAL, str);
#endif
}
using HkvEvictStrategy = nv::merlin::EvictStrategy;

constexpr uint64_t IGNORED_GLOBAL_EPOCH = UINT64_C(0xFFFFFFFFFFFFFFFF);

template <typename K, typename V, typename S>
class KVOnlyFile : public nv::merlin::BaseKVFile<K, V, S> {
 public:
  KVOnlyFile() : keys_fp_(nullptr), values_fp_(nullptr) {}

  ~KVOnlyFile() { close(); }

  bool open(const std::string& keys_path, const std::string& values_path,
            const char* mode) {
    close();
    keys_fp_ = fopen(keys_path.c_str(), mode);
    if (!keys_fp_) {
      return false;
    }
    values_fp_ = fopen(values_path.c_str(), mode);
    if (!values_fp_) {
      close();
      return false;
    }
    return true;
  }

  void close() noexcept {
    if (keys_fp_) {
      fclose(keys_fp_);
      keys_fp_ = nullptr;
    }
    if (values_fp_) {
      fclose(values_fp_);
      values_fp_ = nullptr;
    }
  }

  size_t read(const size_t n, const size_t dim, K* keys, V* vectors,
              S* scores) override {
    size_t nread_keys =
        fread(keys, sizeof(K), static_cast<size_t>(n), keys_fp_);
    size_t nread_vecs =
        fread(vectors, sizeof(V) * dim, static_cast<size_t>(n), values_fp_);
    if (nread_keys != nread_vecs) {
      LOG(INFO) << "Partially read failed. " << nread_keys
                << " kv pairs by KVOnlyFile.";
      return 0;
    }
    LOG(INFO) << "Partially read " << nread_keys << " kv pairs by KVOnlyFile.";
    return nread_keys;
  }

  size_t write(const size_t n, const size_t dim, const K* keys,
               const V* vectors, const S* scores) override {
    size_t nwritten_keys =
        fwrite(keys, sizeof(K), static_cast<size_t>(n), keys_fp_);
    size_t nwritten_vecs =
        fwrite(vectors, sizeof(V) * dim, static_cast<size_t>(n), values_fp_);
    if (nwritten_keys != nwritten_vecs) {
      return 0;
    }
    LOG(INFO) << "Partially write " << nwritten_keys
              << " kv pairs by KVOnlyFile.";
    return nwritten_keys;
  }

 private:
  FILE* keys_fp_;
  FILE* values_fp_;
};

template <typename K, typename V, typename S>
class RandomKVFile : public nv::merlin::BaseKVFile<K, V, S> {
 public:
  RandomKVFile(FileSystem* fs, const std::string& filepath, size_t value_dim,
               size_t buffer_size, bool append_to_file = false)
      : fs_(fs),
        filepath_(filepath),
        value_dim_(value_dim),
        buffer_size_(buffer_size),
        append_to_file_(append_to_file) {}

  ~RandomKVFile() {}

  Status open(const std::string& key_filepath,
              const std::string& value_filepath, const std::string& mode) {
    key_buffer_byte_size_ = buffer_size_ * sizeof(K);
    const size_t value_len = sizeof(V) * value_dim_;
    value_buffer_byte_size_ = buffer_size_ * value_len;

    if ("rb" == mode) {
      TF_RETURN_IF_ERROR(fs_->FileExists(key_filepath));
      TF_RETURN_IF_ERROR(fs_->NewRandomAccessFile(key_filepath, &key_file_));
      key_input_stream_ =
          std::make_unique<io::RandomAccessInputStream>(key_file_.get());
      key_reader_ = std::make_unique<io::BufferedInputStream>(
          key_input_stream_.get(), key_buffer_byte_size_ * 2);

      TF_RETURN_IF_ERROR(fs_->FileExists(value_filepath));
      TF_RETURN_IF_ERROR(
          fs_->NewRandomAccessFile(value_filepath, &value_file_));
      value_input_stream_ =
          std::make_unique<io::RandomAccessInputStream>(value_file_.get());
      value_reader_ = std::make_unique<io::BufferedInputStream>(
          value_input_stream_.get(), value_buffer_byte_size_ * 2);

      uint64 key_file_size = 0;
      TF_RETURN_IF_ERROR(fs_->GetFileSize(key_filepath, &key_file_size));
      size_t key_size = key_file_size / sizeof(K);

      uint64 value_file_size = 0;
      TF_RETURN_IF_ERROR(fs_->GetFileSize(value_filepath, &value_file_size));
      size_t value_size = value_file_size / value_len;

      if (key_size != value_size) {
        return errors::Unavailable(
            "the keys number in file " + key_filepath +
            " is not equal to the value vectors number in file " +
            value_filepath + ".");
      }
    } else if ("wb" == mode) {
      std::string key_tmpfilepath(key_filepath + ".tmp");
      std::string value_tmpfilepath(value_filepath + ".tmp");

      bool has_atomic_move = false;
      auto has_atomic_move_ret =
          fs_->HasAtomicMove(filepath_, &has_atomic_move);
      bool need_tmp_file =
          (has_atomic_move == false) || (has_atomic_move_ret != TFOkStatus);

      if (!need_tmp_file) {
        key_tmpfilepath = key_filepath;
        value_tmpfilepath = value_filepath;
      }
      TF_RETURN_IF_ERROR(
          fs_->RecursivelyCreateDir(std::string(fs_->Dirname(filepath_))));

      if (append_to_file_) {
        TF_RETURN_IF_ERROR(
            fs_->NewAppendableFile(key_tmpfilepath, &key_writer_));
        TF_RETURN_IF_ERROR(
            fs_->NewAppendableFile(value_tmpfilepath, &value_writer_));
      } else {
        TF_RETURN_IF_ERROR(fs_->NewWritableFile(key_tmpfilepath, &key_writer_));
        TF_RETURN_IF_ERROR(
            fs_->NewWritableFile(value_tmpfilepath, &value_writer_));
      }
    }
    return TFOkStatus;
  }

  void close() {
    if (key_writer_) {
      TFRA_LOG_IF_ERROR(key_writer_->Flush());
    }
    if (value_writer_) {
      TFRA_LOG_IF_ERROR(value_writer_->Flush());
    }
  }

  size_t read(const size_t n, const size_t dim, K* keys, V* vectors,
              S* scores) override {
    size_t key_read_byte = n * sizeof(K);
    size_t value_read_byte = n * sizeof(V) * dim;

    key_buffer_.reserve(key_read_byte);
    value_buffer_.reserve(value_read_byte);

    TFRA_LOG_IF_ERROR(key_reader_->ReadNBytes(key_read_byte, &key_buffer_));
    TFRA_LOG_IF_ERROR(
        value_reader_->ReadNBytes(value_read_byte, &value_buffer_));

    memcpy((char*)keys, key_buffer_.data(), key_buffer_.size());
    memcpy((char*)vectors, value_buffer_.data(), value_buffer_.size());

    size_t nread_keys = key_buffer_.size() / sizeof(K);
    return nread_keys;
  }

  size_t write(const size_t n, const size_t dim, const K* keys,
               const V* vectors, const S* scores) override {
    size_t key_write_byte = n * sizeof(K);
    size_t value_write_byte = n * sizeof(V) * value_dim_;

    TFRA_LOG_IF_ERROR(
        key_writer_->Append(StringPiece((char*)keys, key_write_byte)));
    TFRA_LOG_IF_ERROR(
        value_writer_->Append(StringPiece((char*)vectors, value_write_byte)));

    return n;
  }

 private:
  size_t value_dim_;
  FileSystem* fs_ = nullptr;
  std::string filepath_;
  size_t buffer_size_;
  size_t key_buffer_byte_size_;
  size_t value_buffer_byte_size_;
  tstring key_buffer_;
  tstring value_buffer_;
  bool append_to_file_ = false;

  std::unique_ptr<WritableFile> key_writer_ = nullptr;
  std::unique_ptr<WritableFile> value_writer_ = nullptr;

  std::unique_ptr<RandomAccessFile> key_file_ = nullptr;
  std::unique_ptr<RandomAccessFile> value_file_ = nullptr;
  std::unique_ptr<io::RandomAccessInputStream> key_input_stream_ = nullptr;
  std::unique_ptr<io::RandomAccessInputStream> value_input_stream_ = nullptr;
  std::unique_ptr<io::BufferedInputStream> key_reader_ = nullptr;
  std::unique_ptr<io::BufferedInputStream> value_reader_ = nullptr;
};

// template to avoid multidef in compile time only.
template <typename K, typename V>
__global__ void gpu_u64_to_i64_kernel(const uint64_t* u64, int64* i64,
                                      size_t len) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < len) {
    i64[tid] = static_cast<int64>(u64[tid]);
  }
}

template <typename T>
__global__ void broadcast_kernel(T* data, T val, size_t n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    data[tid] = val;
  }
}

template <typename K, typename V>
void gpu_cast_u64_to_i64(const uint64_t* u64, int64* i64, size_t len,
                         cudaStream_t stream) {
  size_t block_size = nv::merlin::SAFE_GET_BLOCK_SIZE(1024);
  size_t grid_size = nv::merlin::SAFE_GET_GRID_SIZE(len, block_size);
  gpu_u64_to_i64_kernel<K, V>
      <<<grid_size, block_size, 0, stream>>>(u64, i64, len);
}

using GPUDevice = Eigen::ThreadPoolDevice;

struct TableWrapperInitOptions {
  size_t max_capacity;
  size_t init_capacity;
  size_t max_hbm_for_vectors;
  size_t max_bucket_size;
  int64_t step_per_epoch;
  int reserved_key_start_bit;

  float max_load_factor;
  int block_size;
  int io_block_size;
};

template <typename V>
__global__ void gpu_fill_default_values(V* d_vals, V* d_def_val, size_t len,
                                        size_t dim) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadId < len) {
#pragma unroll
    for (int i = 0; i < dim; i++) {
      d_vals[threadId * dim + i] = d_def_val[i];
    }
  }
}

class TFOrDefaultAllocator : public nv::merlin::BaseAllocator {
 private:
  using NMMemType = nv::merlin::MemoryType;
  // tensorflow::Allocator* tf_host_allocator_ = nullptr;
  tensorflow::Allocator* tf_device_allocator_ = nullptr;
  std::unique_ptr<nv::merlin::DefaultAllocator> default_allocator_ = nullptr;
  bool use_default_allocator_ = false;
  // bool tf_async_allocator_stream_set_ = false;
  static constexpr size_t kAllocatorAlignment = 4;

 public:
  TFOrDefaultAllocator() : use_default_allocator_(true) {
    default_allocator_ = std::make_unique<nv::merlin::DefaultAllocator>();
  }

  TFOrDefaultAllocator(OpKernelContext* ctx) {
    if (ctx) {
      tensorflow::AllocatorAttributes tf_alloc_attrs;
      tf_device_allocator_ = ctx->get_allocator(tf_alloc_attrs);
    } else {
      use_default_allocator_ = true;
      default_allocator_ = std::make_unique<nv::merlin::DefaultAllocator>();
    }
  }

  ~TFOrDefaultAllocator() override {}

  void alloc(const NMMemType type, void** ptr, size_t size,
             unsigned int pinned_flags = cudaHostAllocDefault) override {
    if (!use_default_allocator_) {
      tensorflow::AllocationAttributes allocation_attr(false, false, nullptr);
      switch (type) {
        case NMMemType::Device:
          *ptr = tf_device_allocator_->AllocateRaw(kAllocatorAlignment, size,
                                                   allocation_attr);
          if (nullptr == *ptr) {
            throw std::runtime_error(
                "Failed to allocator gpu memory, please adjust param 'max_hbm' "
                "smaller.");
          }
          break;
        case NMMemType::Pinned:
          CUDA_CHECK(cudaMallocHost(ptr, size, pinned_flags));
          break;
        case NMMemType::Host:
          *ptr = std::malloc(size);
          break;
      }
    } else {
      default_allocator_->alloc(type, ptr, size, pinned_flags);
    }
  }

  void alloc_async(const NMMemType type, void** ptr, size_t size,
                   cudaStream_t stream) override {
    if (!use_default_allocator_) {
      if (NMMemType::Device == type) {
        *ptr = tf_device_allocator_->AllocateRaw(kAllocatorAlignment, size);
        if (nullptr == *ptr) {
          throw std::runtime_error(
              "Failed to allocator gpu memory, please adjust param 'max_hbm' "
              "smaller.");
        }
      }
    } else {
      default_allocator_->alloc_async(type, ptr, size, stream);
    }
  }

  void free(const NMMemType type, void* ptr) override {
    if (!use_default_allocator_) {
      switch (type) {
        case NMMemType::Device:
          tf_device_allocator_->DeallocateRaw(ptr);
          break;
        case NMMemType::Pinned:
          CUDA_CHECK(cudaFreeHost(ptr));
          break;
        case NMMemType::Host:
          std::free(ptr);
          break;
      }
    } else {
      default_allocator_->free(type, ptr);
    }
  }

  void free_async(const NMMemType type, void* ptr,
                  cudaStream_t stream) override {
    if (!use_default_allocator_) {
      if (NMMemType::Device == type) {
        tf_device_allocator_->DeallocateRaw(ptr);
      }
    } else {
      default_allocator_->free_async(type, ptr, stream);
    }
  }
};

template <class K, class V>
class TableWrapper {
 private:
  using Table = nv::merlin::HashTableBase<K, V, uint64_t>;
  nv::merlin::HashTableOptions mkv_options_;

 public:
  TableWrapper(TableWrapperInitOptions& init_options, size_t dim,
               int strategy) {
    max_capacity_ = init_options.max_capacity;
    dim_ = dim;
    mkv_options_.init_capacity =
        std::min(init_options.init_capacity, max_capacity_);
    mkv_options_.max_capacity = max_capacity_;
    // Since currently GPU nodes are not compatible to fast
    // pcie connections for D2H non-continous wirte, so just
    // use pure hbm mode now.
    // mkv_options_.max_hbm_for_vectors = std::numeric_limits<size_t>::max();
    mkv_options_.max_hbm_for_vectors = init_options.max_hbm_for_vectors;
    mkv_options_.max_load_factor = 0.5;
    mkv_options_.block_size = nv::merlin::SAFE_GET_BLOCK_SIZE(128);
    mkv_options_.dim = dim;

    block_size_ = mkv_options_.block_size;
    switch (strategy) {
      case HkvEvictStrategy::kLfu:
        table_ptr_ = std::make_unique<
            nv::merlin::HashTable<K, V, uint64_t, HkvEvictStrategy::kLfu>>();
        break;
      case HkvEvictStrategy::kEpochLru:
        table_ptr_ = std::make_unique<nv::merlin::HashTable<
            K, V, uint64_t, HkvEvictStrategy::kEpochLru>>();
        break;
      case HkvEvictStrategy::kEpochLfu:
        table_ptr_ = std::make_unique<nv::merlin::HashTable<
            K, V, uint64_t, HkvEvictStrategy::kEpochLfu>>();
        break;
      case HkvEvictStrategy::kCustomized:
        table_ptr_ = std::make_unique<nv::merlin::HashTable<
            K, V, uint64_t, HkvEvictStrategy::kCustomized>>();
        break;
      default:
        table_ptr_ = std::make_unique<
            nv::merlin::HashTable<K, V, uint64_t, HkvEvictStrategy::kLru>>();
        break;
    }
    step_per_epoch_ = init_options.step_per_epoch;
    mkv_options_.reserved_key_start_bit = init_options.reserved_key_start_bit;
    static constexpr size_t default_chunk_buckets = 512;
    size_t min_chunk_buckets = 1;
    for (size_t pow_n = 1; pow_n <= 63; ++pow_n) {
      if (mkv_options_.max_bucket_size * (1 << pow_n) >
          mkv_options_.init_capacity) {
        min_chunk_buckets = 1 << (pow_n - 1);
        break;
      }
    }
    mkv_options_.num_of_buckets_per_alloc =
        mkv_options_.init_capacity >
                (mkv_options_.max_bucket_size * default_chunk_buckets)
            ? default_chunk_buckets
            : min_chunk_buckets;
    curr_epoch_ = 0;
    curr_step_ = 1;

    LOG(INFO) << "Use Evict Strategy:" << strategy
              << ", [0:LRU, 1:LFU, 2:EPOCHLRU, 3:EPOCHLFU, 4:CUSTOMIZED]";
    if (2 == strategy || 3 == strategy) {
      epoch_evict_strategy_ = true;
      table_ptr_->set_global_epoch(curr_epoch_);

      LOG(INFO) << "HKV EPOCH EVICT STRATEGY, step_per_epoch: "
                << init_options.step_per_epoch
                << ", curr_epoch: " << curr_epoch_;
    } else {
      epoch_evict_strategy_ = false;

      table_ptr_->set_global_epoch(IGNORED_GLOBAL_EPOCH);
    }
  }

  ~TableWrapper() {}

  Status init(nv::merlin::BaseAllocator* allocator) {
    try {
      table_ptr_->init(mkv_options_, allocator);
    } catch (std::runtime_error& e) {
      return ReturnInternalErrorStatus(e.what());
    }
    return TFOkStatus;
  }

  void upsert(const K* d_keys, const V* d_vals, const uint64_t* d_scores,
              size_t len, cudaStream_t stream) {
    size_t grid_size = nv::merlin::SAFE_GET_GRID_SIZE(len, block_size_);
    table_ptr_->insert_or_assign(len, d_keys, d_vals, d_scores, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (epoch_evict_strategy_) {
      curr_step_ += 1;
      if (curr_step_ > step_per_epoch_) {
        curr_epoch_ += 1;
        curr_step_ = 1;
        table_ptr_->set_global_epoch(curr_epoch_);
        LOG(INFO) << "HKV EPOCH EVICT STRATEGY: curr_epoch: " << curr_epoch_;
      }
    }
  }

  void accum(const K* d_keys, const V* d_vals_or_deltas, const bool* d_exists,
             const uint64_t* d_scores, size_t len, cudaStream_t stream) {
    uint64_t t0 = (uint64_t)time(NULL);
    size_t grid_size = nv::merlin::SAFE_GET_GRID_SIZE(len, block_size_);
    table_ptr_->accum_or_assign(len, d_keys, d_vals_or_deltas, d_exists,
                                d_scores, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void dump(K* d_key, V* d_val, const size_t offset, const size_t search_length,
            size_t* d_dump_counter, cudaStream_t stream) const {
    table_ptr_->export_batch(search_length, offset, d_dump_counter, d_key,
                             d_val,
                             /*d_scores=*/nullptr, stream);
  }

  void dump_with_scores(K* d_key, V* d_val, uint64_t* d_scores,
                        const size_t offset, const size_t search_length,
                        size_t* d_dump_counter, cudaStream_t stream) const {
    table_ptr_->export_batch(search_length, offset, d_dump_counter, d_key,
                             d_val, d_scores, stream);
  }

  void dump_keys_and_scores(K* keys, int64* scores, size_t len,
                            size_t split_len, cudaStream_t stream) const {
    V* values_buf = nullptr;
    size_t offset = 0;
    size_t real_offset = 0;
    size_t skip = split_len;
    uint64_t* scores_u64 = reinterpret_cast<uint64_t*>(scores);
    size_t span_len = table_ptr_->capacity();
    CUDA_CHECK(
        cudaMallocAsync(&values_buf, sizeof(V) * dim_ * split_len, stream));
    CUDA_CHECK(
        cudaMemsetAsync(values_buf, 0, sizeof(V) * dim_ * split_len, stream));
    for (; offset < span_len; offset += split_len) {
      if (offset + skip > span_len) {
        skip = span_len - offset;
      }
      // TODO: overlap the loop
      size_t h_dump_counter =
          table_ptr_->export_batch(skip, offset, keys + real_offset, values_buf,
                                   scores_u64 + real_offset, stream);
      CudaCheckError();

      if (h_dump_counter > 0) {
        gpu_cast_u64_to_i64<K, V>(scores_u64 + real_offset,
                                  scores + real_offset, h_dump_counter, stream);
        real_offset += h_dump_counter;
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(values_buf, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // TODO (LinGeLin) support scores
  bool is_valid_scores(const std::string& keyfile,
                       const std::string& scorefile) const {
    return false;
  }

  void dump_to_file(FileSystem* fs, const string filepath, size_t dim,
                    cudaStream_t stream, const size_t buffer_size,
                    bool append_to_file) const {
    LOG(INFO) << "dump_to_file, filepath: " << filepath << ", dim: " << dim
              << ", stream: " << stream << ", buffer_size: " << buffer_size;

    std::unique_ptr<nv::merlin::BaseKVFile<K, V, uint64_t>> wfile_ptr;

    string keyfile = filepath + "-keys";
    string valuefile = filepath + "-values";
    string scorefile = filepath + "-scores";
    bool has_scores = false;
    Status status = TFOkStatus;

    if (is_valid_scores(keyfile, scorefile)) {
      wfile_ptr = std::make_unique<nv::merlin::LocalKVFile<K, V, uint64_t>>();
      bool open_ok = reinterpret_cast<nv::merlin::LocalKVFile<K, V, uint64_t>*>(
                         wfile_ptr.get())
                         ->open(keyfile, valuefile, scorefile, "wb");
      has_scores = true;
      if (!open_ok) {
        std::string error_msg = "Failed to dump to file to " + keyfile + ", " +
                                valuefile + ", " + scorefile;
        throw std::runtime_error(error_msg);
      }
    } else {
      wfile_ptr = std::make_unique<RandomKVFile<K, V, uint64_t>>(
          fs, filepath, dim, buffer_size, append_to_file);
      status.Update(
          reinterpret_cast<RandomKVFile<K, V, uint64_t>*>(wfile_ptr.get())
              ->open(keyfile, valuefile, "wb"));
    }
    if (!status.ok()) {
      std::string error_msg = "Failed to dump to file to " + keyfile + ", " +
                              valuefile + ", " + scorefile + " " +
                              status.ToString();
      throw std::runtime_error(error_msg);
    }

    size_t n_saved = table_ptr_->save(
        wfile_ptr.get(),
        buffer_size * (sizeof(K) + sizeof(V) * dim + sizeof(uint64_t)), stream);
    if (has_scores) {
      LOG(INFO) << "[op] Save " << n_saved << " pairs from keyfile: " << keyfile
                << ", and valuefile: " << valuefile << ", and scorefile"
                << scorefile;
    } else {
      LOG(INFO) << "[op] Save " << n_saved << " pairs from keyfile: " << keyfile
                << ", and valuefile: " << valuefile;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (has_scores) {
      reinterpret_cast<nv::merlin::LocalKVFile<K, V, uint64_t>*>(
          wfile_ptr.get())
          ->close();
    } else {
      reinterpret_cast<RandomKVFile<K, V, uint64_t>*>(wfile_ptr.get())->close();
    }
  }

  void load_from_file(FileSystem* fs, const string filepath, size_t dim,
                      cudaStream_t stream, const size_t buffer_size) {
    std::unique_ptr<nv::merlin::BaseKVFile<K, V, uint64_t>> rfile_ptr;
    string keyfile = filepath + "-keys";
    string valuefile = filepath + "-values";
    string scorefile = filepath + "-scores";
    bool has_scores = false;
    Status status = TFOkStatus;

    if (is_valid_scores(keyfile, scorefile)) {
      rfile_ptr = std::make_unique<nv::merlin::LocalKVFile<K, V, uint64_t>>();
      bool open_ok = reinterpret_cast<nv::merlin::LocalKVFile<K, V, uint64_t>*>(
                         rfile_ptr.get())
                         ->open(keyfile, valuefile, scorefile, "rb");
      has_scores = true;
      if (!open_ok) {
        std::string error_msg = "Failed to load from file " + keyfile + ", " +
                                valuefile + ", " + scorefile;
        throw std::runtime_error(error_msg);
      }
    } else {
      rfile_ptr = std::make_unique<RandomKVFile<K, V, uint64_t>>(
          fs, filepath, dim, buffer_size);
      status.Update(
          reinterpret_cast<RandomKVFile<K, V, uint64_t>*>(rfile_ptr.get())
              ->open(keyfile, valuefile, "rb"));
    }
    if (!status.ok()) {
      std::string error_msg = "Failed to load from file " + keyfile + ", " +
                              valuefile + ", " + scorefile + " " +
                              status.ToString();
      throw std::runtime_error(error_msg);
    }

    size_t n_loaded = table_ptr_->load(
        rfile_ptr.get(),
        buffer_size * (sizeof(K) + sizeof(V) * dim + sizeof(uint64_t)), stream);
    if (has_scores) {
      LOG(INFO) << "[op] Load " << n_loaded
                << " pairs from keyfile: " << keyfile
                << ", and valuefile: " << valuefile << ", and scorefile"
                << scorefile;
    } else {
      LOG(INFO) << "[op] Load " << n_loaded
                << " pairs from keyfile: " << keyfile
                << ", and valuefile: " << valuefile;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (has_scores) {
      reinterpret_cast<nv::merlin::LocalKVFile<K, V, uint64_t>*>(
          rfile_ptr.get())
          ->close();
    } else {
      reinterpret_cast<RandomKVFile<K, V, uint64_t>*>(rfile_ptr.get())->close();
    }
  }

  void get(const K* d_keys, V* d_vals, bool* d_status, size_t len, V* d_def_val,
           cudaStream_t stream, bool is_full_size_default) const {
    if (is_full_size_default) {
      CUDA_CHECK(cudaMemcpyAsync(d_vals, d_def_val, sizeof(V) * dim_ * len,
                                 cudaMemcpyDeviceToDevice, stream));
    } else {
      size_t grid_size = nv::merlin::SAFE_GET_GRID_SIZE(len, block_size_);
      gpu_fill_default_values<V>
          <<<grid_size, block_size_, dim_ * sizeof(V), stream>>>(
              d_vals, d_def_val, len, dim_);
    }
    table_ptr_->find(len, d_keys, d_vals, d_status, /*d_scores=*/nullptr,
                     stream);
  }

  // TODO(LinGeLin): Implemented using the HKV contains API
  void contains(const K* d_keys, V* d_status, size_t len, cudaStream_t stream) {
    // pass
    V* tmp_vals = nullptr;
    CUDA_CHECK(cudaMallocAsync(&tmp_vals, sizeof(V) * len * dim_, stream));
    CUDA_CHECK(cudaMemsetAsync(&tmp_vals, 0, sizeof(V) * len * dim_, stream));
    table_ptr_->find(len, d_keys, tmp_vals, d_status, /*d_scores=*/nullptr,
                     stream);
    CUDA_CHECK(cudaFreeAsync(tmp_vals, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  size_t get_size(cudaStream_t stream) const {
    return table_ptr_->size(stream);
  }

  size_t get_capacity() const { return table_ptr_->capacity(); }

  void remove(const K* d_keys, size_t len, cudaStream_t stream) {
    table_ptr_->erase(len, d_keys, stream);
  }

  void clear(cudaStream_t stream) { table_ptr_->clear(stream); }

 private:
  std::unique_ptr<Table> table_ptr_;
  size_t max_capacity_;
  size_t dim_;
  int64_t step_per_epoch_;
  int64_t curr_step_;
  uint64_t curr_epoch_;
  int block_size_;
  bool dynamic_mode_;
  bool epoch_evict_strategy_;
};

template <class K, class V>
Status CreateTableImpl(TableWrapper<K, V>** pptable,
                       TableWrapperInitOptions& options,
                       nv::merlin::BaseAllocator* allocator, size_t runtime_dim,
                       int strategy) {
  *pptable = new TableWrapper<K, V>(options, runtime_dim, strategy);
  return (*pptable)->init(allocator);
}

}  // namespace gpu
}  // namespace hkv_table
}  // namespace recommenders_addons
}  // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_HKV_H_
