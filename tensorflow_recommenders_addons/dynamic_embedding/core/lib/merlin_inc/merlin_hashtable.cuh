/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include "merlin/array_kernels.cuh"
#include "merlin/core_kernels.cuh"
#include "merlin/flexible_buffer.cuh"
#include "merlin/group_lock.hpp"
#include "merlin/memory_pool.cuh"
#include "merlin/types.cuh"
#include "merlin/utils.cuh"

namespace nv {
namespace merlin {

/**
 * @brief Enumeration of the eviction strategies.
 *
 * @note The `meta` is introduced to define the importance of each key, the
 * larger, the more important, the less likely they will be evicted. On `kLru`
 * mode, the `metas` parameter of the APIs should keep `nullptr`, the meta for
 * each key is assigned internally in LRU(Least Recently Used) policy. On
 * `kCustomized` mode, the `metas` should be provided by caller.
 *
 * @note Eviction occurs automatically when a bucket is full. The keys with the
 * minimum `meta` value are evicted first.
 *
 */
enum class EvictStrategy {
  kLru = 0,        ///< LRU mode.
  kCustomized = 1  ///< Customized mode.
};

/**
 * @brief The options struct of HierarchicalKV.
 */
struct HashTableOptions {
  size_t init_capacity = 0;        ///< The initial capacity of the hash table.
  size_t max_capacity = 0;         ///< The maximum capacity of the hash table.
  size_t max_hbm_for_vectors = 0;  ///< The maximum HBM for vectors, in bytes.
  size_t max_bucket_size = 128;    ///< The length of each bucket.
  size_t dim = 64;                 ///< The dimension of the vectors.
  float max_load_factor = 0.5f;    ///< The max load factor before rehashing.
  int block_size = 128;            ///< The default block size for CUDA kernels.
  int io_block_size = 1024;        ///< The block size for IO CUDA kernels.
  int device_id = -1;              ///< The ID of device.
  bool io_by_cpu = false;  ///< The flag indicating if the CPU handles IO.
  EvictStrategy evict_strategy = EvictStrategy::kLru;  ///< The evict strategy.
  bool use_constant_memory = false;                    ///< reserved
  MemoryPoolOptions
      device_memory_pool;  ///< Configuration options for device memory pool.
  MemoryPoolOptions
      host_memory_pool;  ///< Configuration options for host memory pool.
};

/**
 * @brief A customizable template function indicates which keys should be
 * erased from the hash table by returning `true`.
 *
 * @note The `erase_if` or `export_batch_if` API traverses all of the items by
 * this function and the items that return `true` are removed or exported.
 *
 *  Example for erase_if:
 *
 *    ```
 *    template <class K, class M>
 *    __forceinline__ __device__ bool erase_if_pred(const K& key,
 *                                                  M& meta,
 *                                                  const K& pattern,
 *                                                  const M& threshold) {
 *      return ((key & 0xFFFF000000000000 == pattern) &&
 *              (meta < threshold));
 *    }
 *    ```
 *
 *  Example for export_batch_if:
 *    ```
 *    template <class K, class M>
 *    __forceinline__ __device__ bool export_if_pred(const K& key,
 *                                                   M& meta,
 *                                                   const K& pattern,
 *                                                   const M& threshold) {
 *      return meta >= threshold;
 *    }
 *    ```
 */
template <class K, class M>
using EraseIfPredict = bool (*)(
    const K& key,       ///< The traversed key in a hash table.
    M& meta,            ///< The traversed meta in a hash table.
    const K& pattern,   ///< The key pattern to compare with the `key` argument.
    const M& threshold  ///< The threshold to compare with the `meta` argument.
);

/**
 * A HierarchicalKV hash table is a concurrent and hierarchical hash table that
 * is powered by GPUs and can use HBM and host memory as storage for key-value
 * pairs. Support for SSD storage is a future consideration.
 *
 * The `meta` is introduced to define the importance of each key, the
 * larger, the more important, the less likely they will be evicted. Eviction
 * occurs automatically when a bucket is full. The keys with the minimum `meta`
 * value are evicted first. In a customized eviction strategy, we recommend
 * using the timestamp or frequency of the key occurrence as the `meta` value
 * for each key. You can also assign a special value to the `meta` to
 * perform a customized eviction strategy.
 *
 * @note By default configuration, this class is thread-safe.
 *
 * @tparam K The data type of the key.
 * @tparam V The data type of the vector's item type.
 *         The item data type should be a basic data type of C++/CUDA.
 * @tparam M The data type for `meta`.
 *           The currently supported data type is only `uint64_t`.
 *
 */
template <class K, class V, class M = uint64_t>
class HashTable {
 public:
  using size_type = size_t;
  using key_type = K;
  using value_type = V;
  using meta_type = M;
  using Pred = EraseIfPredict<key_type, meta_type>;

 private:
  using TableCore = nv::merlin::Table<key_type, value_type, meta_type>;
  static constexpr unsigned int TILE_SIZE = 4;

  using DeviceMemoryPool = MemoryPool<DeviceAllocator<char>>;
  using HostMemoryPool = MemoryPool<HostAllocator<char>>;

#if THRUST_VERSION >= 101600
  static constexpr auto thrust_par = thrust::cuda::par_nosync;
#else
  static constexpr auto thrust_par = thrust::cuda::par;
#endif

 public:
  /**
   * @brief Default constructor for the hash table class.
   */
  HashTable(){};

  /**
   * @brief Frees the resources used by the hash table and destroys the hash
   * table object.
   */
  ~HashTable() {
    if (initialized_) {
      CUDA_CHECK(cudaDeviceSynchronize());

      initialized_ = false;
      destroy_table<key_type, value_type, meta_type>(&table_);
      CUDA_CHECK(cudaFree(d_table_));
      dev_mem_pool_.reset();
      host_mem_pool_.reset();
    }
  }

 private:
  HashTable(const HashTable&) = delete;
  HashTable& operator=(const HashTable&) = delete;
  HashTable(HashTable&&) = delete;
  HashTable& operator=(HashTable&&) = delete;

 public:
  /**
   * @brief Initialize a merlin::HashTable.
   *
   * @param options The configuration options.
   */
  void init(const HashTableOptions options) {
    if (initialized_) {
      return;
    }
    options_ = options;

    if (options_.device_id >= 0) {
      CUDA_CHECK(cudaSetDevice(options_.device_id));
    } else {
      CUDA_CHECK(cudaGetDevice(&(options_.device_id)));
    }

    // Construct table.
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, options_.device_id));
    shared_mem_size_ = deviceProp.sharedMemPerBlock;
    create_table<key_type, value_type, meta_type>(
        &table_, options_.dim, options_.init_capacity, options_.max_capacity,
        options_.max_hbm_for_vectors, options_.max_bucket_size);
    options_.block_size = SAFE_GET_BLOCK_SIZE(options_.block_size);
    reach_max_capacity_ = (options_.init_capacity * 2 > options_.max_capacity);
    MERLIN_CHECK((!(options_.io_by_cpu && options_.max_hbm_for_vectors != 0)),
                 "[HierarchicalKV] `io_by_cpu` should not be true when "
                 "`max_hbm_for_vectors` is not 0!");
    CUDA_CHECK(cudaMalloc((void**)&(d_table_), sizeof(TableCore)));

    sync_table_configuration();

    // Create memory pools.
    dev_mem_pool_ = std::make_unique<MemoryPool<DeviceAllocator<char>>>(
        options_.device_memory_pool);
    host_mem_pool_ = std::make_unique<MemoryPool<HostAllocator<char>>>(
        options_.host_memory_pool);

    CUDA_CHECK(cudaDeviceSynchronize());
    initialized_ = true;
    CudaCheckError();
  }

  /**
   * @brief Insert new key-value-meta tuples into the hash table.
   * If the key already exists, the values and metas are assigned new values.
   *
   * If the target bucket is full, the keys with minimum meta will be
   * overwritten by new key unless the meta of the new key is even less than
   * minimum meta of the target bucket.
   *
   * @param n Number of key-value-meta tuples to insert or assign.
   * @param keys The keys to insert on GPU-accessible memory with shape
   * (n).
   * @param values The values to insert on GPU-accessible memory with
   * shape (n, DIM).
   * @param metas The metas to insert on GPU-accessible memory with shape
   * (n).
   * @parblock
   * The metas should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p metas should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the insert_or_assign ignores the evict strategy of table with current
   * metas anyway. If true, it does not check whether the metas confroms to
   * the evict strategy. If false, it requires the metas follow the evict
   * strategy of table.
   */
  void insert_or_assign(const size_type n,
                        const key_type* keys,              // (n)
                        const value_type* values,          // (n, DIM)
                        const meta_type* metas = nullptr,  // (n)
                        cudaStream_t stream = 0,
                        bool ignore_evict_strategy = false) {
    if (n == 0) {
      return;
    }

    while (!reach_max_capacity_ &&
           fast_load_factor(n, stream) > options_.max_load_factor) {
      reserve(capacity() * 2, stream);
    }

    if (!ignore_evict_strategy) {
      check_evict_strategy(metas);
    }

    writer_shared_lock lock(mutex_);

    if (is_fast_mode()) {
      using Selector =
          SelectUpsertKernelWithIO<key_type, value_type, meta_type>;
      static thread_local int step_counter = 0;
      static thread_local float load_factor = 0.0;

      if (((step_counter++) % kernel_select_interval_) == 0) {
        load_factor = fast_load_factor(0, stream, false);
      }

      Selector::execute_kernel(
          load_factor, options_.block_size, options_.max_bucket_size,
          table_->buckets_num, options_.dim, stream, n, d_table_, keys,
          reinterpret_cast<const value_type*>(values), metas);
    } else {
      const size_type dev_ws_size{n * (sizeof(value_type*) + sizeof(int))};
      auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
      auto d_dst{dev_ws.get<value_type**>(0)};
      auto d_src_offset{reinterpret_cast<int*>(d_dst + n)};

      CUDA_CHECK(cudaMemsetAsync(d_dst, 0, dev_ws_size, stream));

      {
        const size_t block_size = options_.block_size;
        const size_t N = n * TILE_SIZE;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        upsert_kernel<key_type, value_type, meta_type, TILE_SIZE>
            <<<grid_size, block_size, 0, stream>>>(
                d_table_, options_.max_bucket_size, table_->buckets_num,
                options_.dim, keys, d_dst, metas, d_src_offset, N);
      }

      {
        thrust::device_ptr<uintptr_t> d_dst_ptr(
            reinterpret_cast<uintptr_t*>(d_dst));
        thrust::device_ptr<int> d_src_offset_ptr(d_src_offset);

        thrust::sort_by_key(thrust_par.on(stream), d_dst_ptr, d_dst_ptr + n,
                            d_src_offset_ptr, thrust::less<uintptr_t>());
      }

      if (options_.io_by_cpu) {
        const size_type host_ws_size{dev_ws_size +
                                     n * sizeof(value_type) * dim()};
        auto host_ws{host_mem_pool_->get_workspace<1>(host_ws_size, stream)};
        auto h_dst{host_ws.get<value_type**>(0)};
        auto h_src_offset{reinterpret_cast<int*>(h_dst + n)};
        auto h_values{reinterpret_cast<value_type*>(h_src_offset + n)};

        CUDA_CHECK(cudaMemcpyAsync(h_dst, d_dst, dev_ws_size,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_values, values, host_ws_size - dev_ws_size,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        write_by_cpu<value_type>(h_dst, h_values, h_src_offset, dim(), n);
      } else {
        const size_t block_size = options_.io_block_size;
        const size_t N = n * dim();
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        write_kernel<key_type, value_type, meta_type>
            <<<grid_size, block_size, 0, stream>>>(values, d_dst, d_src_offset,
                                                   dim(), N);
      }
    }

    CudaCheckError();
  }

  /**
   * @brief Insert new key-value-meta tuples into the hash table.
   * If the key already exists, the values and metas are assigned new values.
   *
   * If the target bucket is full, the keys with minimum meta will be
   * overwritten by new key unless the meta of the new key is even less than
   * minimum meta of the target bucket. The overwritten key with minimum
   * meta will be evicted, with its values and meta, to evicted_keys,
   * evicted_values, evcted_metas seperately in compact format.
   *
   * @param n Number of key-value-meta tuples to insert or assign.
   * @param keys The keys to insert on GPU-accessible memory with shape
   * (n).
   * @param values The values to insert on GPU-accessible memory with
   * shape (n, DIM).
   * @param metas The metas to insert on GPU-accessible memory with shape
   * (n).
   * @param metas The metas to insert on GPU-accessible memory with shape
   * (n).
   * @params evicted_keys The output of keys replaced with minimum meta.
   * @params evicted_values The output of values replaced with minimum meta on
   * keys.
   * @params evicted_metas The output of metas replaced with minimum meta on
   * keys.
   * @parblock
   * The metas should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p metas should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the insert_or_assign ignores the evict strategy of table with current
   * metas anyway. If true, it does not check whether the metas confroms to
   * the evict strategy. If false, it requires the metas follow the evict
   * strategy of table.
   */
  size_type insert_and_evict(const size_type n,
                             const key_type* keys,        // (n)
                             const value_type* values,    // (n, DIM)
                             const meta_type* metas,      // (n)
                             key_type* evicted_keys,      // (n)
                             value_type* evicted_values,  // (n, DIM)
                             meta_type* evicted_metas,    // (n)
                             cudaStream_t stream = 0) {
    if (n == 0) {
      return 0;
    }

    while (!reach_max_capacity_ &&
           fast_load_factor(n, stream) > options_.max_load_factor) {
      reserve(capacity() * 2, stream);
    }

    writer_shared_lock lock(mutex_);

    // TODO: Currently only need eviction when using HashTable as HBM cache.
    if (!is_fast_mode()) {
      throw std::runtime_error("Only allow insert_and_evict in pure HBM mode.");
    }

    using Selector =
        SelectUpsertAndEvictKernelWithIO<key_type, value_type, meta_type>;
    static thread_local int step_counter = 0;
    static thread_local float load_factor = 0.0;

    if (((step_counter++) % kernel_select_interval_) == 0) {
      load_factor = fast_load_factor(0, stream, false);
    }

    // always use max tile to avoid data-deps as possible.
    const int TILE_SIZE = 32;
    size_t n_offsets = (n + TILE_SIZE - 1) / TILE_SIZE;
    const size_type dev_ws_size =
        n_offsets * sizeof(int64_t) + n * sizeof(bool) + sizeof(size_type);

    auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
    auto d_offsets{dev_ws.get<int64_t*>(0)};
    auto dn_evicted = reinterpret_cast<size_type*>(d_offsets + n_offsets);
    auto d_masks = reinterpret_cast<bool*>(dn_evicted + 1);

    CUDA_CHECK(
        cudaMemsetAsync(d_offsets, 0, n_offsets * sizeof(int64_t), stream));
    CUDA_CHECK(cudaMemsetAsync(dn_evicted, 0, sizeof(size_type), stream));
    CUDA_CHECK(cudaMemsetAsync(d_masks, 0, n * sizeof(bool), stream));

    size_type block_size = options_.block_size;
    size_type grid_size = SAFE_GET_GRID_SIZE(n, block_size);
    CUDA_CHECK(cudaMemsetAsync(evicted_keys, static_cast<int>(EMPTY_KEY),
                               n * sizeof(K), stream));

    Selector::execute_kernel(
        load_factor, options_.block_size, options_.max_bucket_size,
        table_->buckets_num, options_.dim, stream, n, d_table_, keys, values,
        metas, evicted_keys, evicted_values, evicted_metas);

    keys_not_empty<K>
        <<<grid_size, block_size, 0, stream>>>(evicted_keys, d_masks, n);
    size_type n_evicted = 0;
    gpu_pick_kvm_inplace<K, V, M, int64_t, TILE_SIZE>(
        grid_size, block_size, d_masks, true, n, dn_evicted, d_offsets, evicted_keys,
        evicted_values, evicted_metas, dim(), stream);
    CUDA_CHECK(cudaMemcpyAsync(&n_evicted, dn_evicted, sizeof(size_type),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CudaCheckError();
    return n_evicted;
  }

  /**
   * Searches for each key in @p keys in the hash table.
   * If the key is found and the corresponding value in @p accum_or_assigns is
   * `true`, the @p vectors_or_deltas is treated as a delta to the old
   * value, and the delta is added to the old value of the key.
   *
   * If the key is not found and the corresponding value in @p accum_or_assigns
   * is `false`, the @p vectors_or_deltas is treated as a new value and the
   * key-value pair is updated in the table directly.
   *
   * @note When the key is found and the value of @p accum_or_assigns is
   * `false`, or when the key is not found and the value of @p accum_or_assigns
   * is `true`, nothing is changed and this operation is ignored.
   * The algorithm assumes these situations occur while the key was modified or
   * removed by other processes just now.
   *
   * @param n The number of key-value-meta tuples to process.
   * @param keys The keys to insert on GPU-accessible memory with shape (n).
   * @param value_or_deltas The values or deltas to insert on GPU-accessible
   * memory with shape (n, DIM).
   * @param accum_or_assigns The operation type with shape (n). A value of
   * `true` indicates to accum and `false` indicates to assign.
   * @param metas The metas to insert on GPU-accessible memory with shape (n).
   * @parblock
   * The metas should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p metas should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the accum_or_assign ignores the evict strategy of table with current
   * metas anyway. If true, it does not check whether the metas confroms to
   * the evict strategy. If false, it requires the metas follow the evict
   * strategy of table.
   *
   */
  void accum_or_assign(const size_type n,
                       const key_type* keys,               // (n)
                       const value_type* value_or_deltas,  // (n, DIM)
                       const bool* accum_or_assigns,       // (n)
                       const meta_type* metas = nullptr,   // (n)
                       cudaStream_t stream = 0,
                       bool ignore_evict_strategy = false) {
    if (n == 0) {
      return;
    }

    while (!reach_max_capacity_ &&
           fast_load_factor(n, stream) > options_.max_load_factor) {
      reserve(capacity() * 2, stream);
    }

    if (!ignore_evict_strategy) {
      check_evict_strategy(metas);
    }

    writer_shared_lock lock(mutex_);

    const size_type dev_ws_size{
        n * (sizeof(value_type*) + sizeof(int) + sizeof(bool))};
    auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
    auto dst{dev_ws.get<value_type**>(0)};
    auto src_offset{reinterpret_cast<int*>(dst + n)};
    auto founds{reinterpret_cast<bool*>(src_offset + n)};

    CUDA_CHECK(cudaMemsetAsync(dst, 0, dev_ws_size, stream));

    {
      const size_t block_size = options_.block_size;
      const size_t N = n * TILE_SIZE;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      accum_kernel<key_type, value_type, meta_type>
          <<<grid_size, block_size, 0, stream>>>(
              table_, keys, dst, metas, accum_or_assigns, table_->buckets,
              table_->buckets_size, table_->bucket_max_size,
              table_->buckets_num, src_offset, founds, N);
    }

    if (!is_fast_mode()) {
      thrust::device_ptr<uintptr_t> dst_ptr(reinterpret_cast<uintptr_t*>(dst));
      thrust::device_ptr<int> src_offset_ptr(src_offset);

      thrust::sort_by_key(thrust_par.on(stream), dst_ptr, dst_ptr + n,
                          src_offset_ptr, thrust::less<uintptr_t>());
    }

    {
      const size_t block_size = options_.io_block_size;
      const size_t N = n * dim();
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      write_with_accum_kernel<key_type, value_type, meta_type>
          <<<grid_size, block_size, 0, stream>>>(value_or_deltas, dst,
                                                 accum_or_assigns, founds,
                                                 src_offset, dim(), N);
    }

    CudaCheckError();
  }

  /**
   * @brief Searches the hash table for the specified keys.
   * When a key is missing, the value in @p values and @p metas will be
   * inserted.
   *
   * @param n The number of key-value-meta tuples to search or insert.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param values The values to search on GPU-accessible memory with
   * shape (n, DIM).
   * @param metas The metas to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p metas is `nullptr`, the meta for each key will not be returned.
   * @endparblock
   * @param stream The CUDA stream that is used to execute the operation.
   *
   */
  void find_or_insert(const size_type n, const key_type* keys,  // (n)
                      value_type* values,                       // (n * DIM)
                      meta_type* metas = nullptr,               // (n)
                      cudaStream_t stream = 0,
                      bool ignore_evict_strategy = false) {
    if (n == 0) {
      return;
    }

    while (!reach_max_capacity_ &&
           fast_load_factor(n, stream) > options_.max_load_factor) {
      reserve(capacity() * 2, stream);
    }

    if (!ignore_evict_strategy) {
      check_evict_strategy(metas);
    }

    writer_shared_lock lock(mutex_);

    if (is_fast_mode()) {
      using Selector =
          SelectFindOrInsertKernelWithIO<key_type, value_type, meta_type>;
      static thread_local int step_counter = 0;
      static thread_local float load_factor = 0.0;

      if (((step_counter++) % kernel_select_interval_) == 0) {
        load_factor = fast_load_factor(0, stream, false);
      }
      Selector::execute_kernel(load_factor, options_.block_size,
                               options_.max_bucket_size, table_->buckets_num,
                               options_.dim, stream, n, d_table_, keys, values,
                               metas);
    } else {
      const size_type dev_ws_size{
          n * (sizeof(value_type*) + sizeof(int) + sizeof(bool))};
      auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
      auto d_table_value_addrs{dev_ws.get<value_type**>(0)};
      auto param_key_index{reinterpret_cast<int*>(d_table_value_addrs + n)};
      auto founds{reinterpret_cast<bool*>(param_key_index + n)};

      CUDA_CHECK(cudaMemsetAsync(d_table_value_addrs, 0, dev_ws_size, stream));

      {
        const size_t block_size = options_.block_size;
        const size_t N = n * TILE_SIZE;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        find_or_insert_kernel<key_type, value_type, meta_type, TILE_SIZE>
            <<<grid_size, block_size, 0, stream>>>(
                d_table_, options_.max_bucket_size, table_->buckets_num,
                options_.dim, keys, d_table_value_addrs, metas, founds,
                param_key_index, N);
      }

      {
        thrust::device_ptr<uintptr_t> table_value_ptr(
            reinterpret_cast<uintptr_t*>(d_table_value_addrs));
        thrust::device_ptr<int> param_key_index_ptr(param_key_index);

        thrust::sort_by_key(thrust_par.on(stream), table_value_ptr,
                            table_value_ptr + n, param_key_index_ptr,
                            thrust::less<uintptr_t>());
      }

      if (options_.io_by_cpu) {
        const size_type host_ws_size{
            dev_ws_size + n * (sizeof(bool) + sizeof(value_type) * dim())};
        auto host_ws{host_mem_pool_->get_workspace<1>(host_ws_size, stream)};
        auto h_table_value_addrs{host_ws.get<value_type**>(0)};
        auto h_param_key_index{reinterpret_cast<int*>(h_table_value_addrs + n)};
        auto h_founds{reinterpret_cast<bool*>(h_param_key_index + n)};
        auto h_param_values{reinterpret_cast<value_type*>(h_founds + n)};

        CUDA_CHECK(cudaMemcpyAsync(h_table_value_addrs, d_table_value_addrs,
                                   dev_ws_size, cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(h_founds, founds, n * sizeof(bool),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_param_values, values,
                                   n * sizeof(value_type) * dim(),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        read_or_write_by_cpu<value_type>(h_table_value_addrs, h_param_values,
                                         h_param_key_index, h_founds, dim(), n);
        CUDA_CHECK(cudaMemcpyAsync(values, h_param_values,
                                   n * sizeof(value_type) * dim(),
                                   cudaMemcpyHostToDevice, stream));
      } else {
        const size_t block_size = options_.io_block_size;
        const size_t N = n * dim();
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        read_or_write_kernel<key_type, value_type, meta_type>
            <<<grid_size, block_size, 0, stream>>>(
                d_table_value_addrs, values, founds, param_key_index, dim(), N);
      }
    }

    CudaCheckError();
  }

  /**
   * @brief Searches the hash table for the specified keys and returns address
   * of the values. When a key is missing, the value in @p values and @p metas
   * will be inserted.
   *
   * @warning This API returns internal addresses for high-performance but
   * thread-unsafe. The caller is responsible for guaranteeing data consistency.
   *
   * @param n The number of key-value-meta tuples to search or insert.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param values  The addresses of values to search on GPU-accessible memory
   * with shape (n).
   * @param founds The status that indicates if the keys are found on
   * @param metas The metas to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p metas is `nullptr`, the meta for each key will not be returned.
   * @endparblock
   * @param stream The CUDA stream that is used to execute the operation.
   *
   */
  void find_or_insert(const size_type n, const key_type* keys,  // (n)
                      value_type** values,                      // (n)
                      bool* founds,                             // (n)
                      meta_type* metas = nullptr,               // (n)
                      cudaStream_t stream = 0,
                      bool ignore_evict_strategy = false) {
    if (n == 0) {
      return;
    }

    while (!reach_max_capacity_ &&
           fast_load_factor(n, stream) > options_.max_load_factor) {
      reserve(capacity() * 2, stream);
    }

    if (!ignore_evict_strategy) {
      check_evict_strategy(metas);
    }

    writer_shared_lock lock(mutex_);

    using Selector =
        SelectFindOrInsertPtrKernel<key_type, value_type, meta_type>;
    static thread_local int step_counter = 0;
    static thread_local float load_factor = 0.0;

    if (((step_counter++) % kernel_select_interval_) == 0) {
      load_factor = fast_load_factor(0, stream, false);
    }
    Selector::execute_kernel(load_factor, options_.block_size,
                             options_.max_bucket_size, table_->buckets_num,
                             options_.dim, stream, n, d_table_, keys, values,
                             metas, founds);

    CudaCheckError();
  }
  /**
   * @brief Assign new key-value-meta tuples into the hash table.
   * If the key doesn't exist, the operation on the key will be ignored.
   *
   * @param n Number of key-value-meta tuples to insert or assign.
   * @param keys The keys to insert on GPU-accessible memory with shape
   * (n).
   * @param values The values to insert on GPU-accessible memory with
   * shape (n, DIM).
   * @param metas The metas to insert on GPU-accessible memory with shape
   * (n).
   * @parblock
   * The metas should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p metas should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   */
  void assign(const size_type n,
              const key_type* keys,              // (n)
              const value_type* values,          // (n, DIM)
              const meta_type* metas = nullptr,  // (n)
              cudaStream_t stream = 0) {
    if (n == 0) {
      return;
    }

    writer_shared_lock lock(mutex_);

    if (is_fast_mode()) {
      using Selector =
          SelectUpdateKernelWithIO<key_type, value_type, meta_type>;
      static thread_local int step_counter = 0;
      static thread_local float load_factor = 0.0;

      if (((step_counter++) % kernel_select_interval_) == 0) {
        load_factor = fast_load_factor(0, stream, false);
      }

      Selector::execute_kernel(load_factor, options_.block_size,
                               options_.max_bucket_size, table_->buckets_num,
                               options_.dim, stream, n, d_table_, keys, values,
                               metas);
    } else {
      const size_type dev_ws_size{n * (sizeof(value_type*) + sizeof(int))};
      auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
      auto d_dst{dev_ws.get<value_type**>(0)};
      auto d_src_offset{reinterpret_cast<int*>(d_dst + n)};

      CUDA_CHECK(cudaMemsetAsync(d_dst, 0, dev_ws_size, stream));

      {
        const size_t block_size = options_.block_size;
        const size_t N = n * TILE_SIZE;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        update_kernel<key_type, value_type, meta_type, TILE_SIZE>
            <<<grid_size, block_size, 0, stream>>>(
                d_table_, options_.max_bucket_size, table_->buckets_num,
                options_.dim, keys, d_dst, metas, d_src_offset, N);
      }

      {
        thrust::device_ptr<uintptr_t> d_dst_ptr(
            reinterpret_cast<uintptr_t*>(d_dst));
        thrust::device_ptr<int> d_src_offset_ptr(d_src_offset);

        thrust::sort_by_key(thrust_par.on(stream), d_dst_ptr, d_dst_ptr + n,
                            d_src_offset_ptr, thrust::less<uintptr_t>());
      }

      if (options_.io_by_cpu) {
        const size_type host_ws_size{dev_ws_size +
                                     n * sizeof(value_type) * dim()};
        auto host_ws{host_mem_pool_->get_workspace<1>(host_ws_size, stream)};
        auto h_dst{host_ws.get<value_type**>(0)};
        auto h_src_offset{reinterpret_cast<int*>(h_dst + n)};
        auto h_values{reinterpret_cast<value_type*>(h_src_offset + n)};

        CUDA_CHECK(cudaMemcpyAsync(h_dst, d_dst, dev_ws_size,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_values, values, host_ws_size - dev_ws_size,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        write_by_cpu<value_type>(h_dst, h_values, h_src_offset, dim(), n);
      } else {
        const size_t block_size = options_.io_block_size;
        const size_t N = n * dim();
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        write_kernel<key_type, value_type, meta_type>
            <<<grid_size, block_size, 0, stream>>>(values, d_dst, d_src_offset,
                                                   dim(), N);
      }
    }

    CudaCheckError();
  }

  /**
   * @brief Searches the hash table for the specified keys.
   *
   * @note When a key is missing, the value in @p values is not changed.
   *
   * @param n The number of key-value-meta tuples to search.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param values The values to search on GPU-accessible memory with
   * shape (n, DIM).
   * @param founds The status that indicates if the keys are found on
   * GPU-accessible memory with shape (n).
   * @param metas The metas to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p metas is `nullptr`, the meta for each key will not be returned.
   * @endparblock
   * @param stream The CUDA stream that is used to execute the operation.
   *
   */
  void find(const size_type n, const key_type* keys,  // (n)
            value_type* values,                       // (n, DIM)
            bool* founds,                             // (n)
            meta_type* metas = nullptr,               // (n)
            cudaStream_t stream = 0) const {
    if (n == 0) {
      return;
    }

    CUDA_CHECK(cudaMemsetAsync(founds, 0, n * sizeof(bool), stream));

    reader_shared_lock lock(mutex_);

    if (is_fast_mode()) {
      using Selector =
          SelectLookupKernelWithIO<key_type, value_type, meta_type>;
      static thread_local int step_counter = 0;
      static thread_local float load_factor = 0.0;

      if (((step_counter++) % kernel_select_interval_) == 0) {
        load_factor = fast_load_factor(0, stream, false);
      }
      Selector::execute_kernel(load_factor, options_.block_size,
                               options_.max_bucket_size, table_->buckets_num,
                               options_.dim, stream, n, d_table_, keys, values,
                               metas, founds);
    } else {
      const size_type dev_ws_size{n * (sizeof(value_type*) + sizeof(int))};
      auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
      auto src{dev_ws.get<value_type**>(0)};
      auto dst_offset{reinterpret_cast<int*>(src + n)};

      CUDA_CHECK(cudaMemsetAsync(src, 0, dev_ws_size, stream));

      {
        const size_t block_size = options_.block_size;
        const size_t N = n * TILE_SIZE;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        lookup_kernel<key_type, value_type, meta_type, TILE_SIZE>
            <<<grid_size, block_size, 0, stream>>>(
                d_table_, options_.max_bucket_size, table_->buckets_num,
                options_.dim, keys, src, metas, founds, dst_offset, N);
      }

      {
        thrust::device_ptr<uintptr_t> src_ptr(
            reinterpret_cast<uintptr_t*>(src));
        thrust::device_ptr<int> dst_offset_ptr(dst_offset);

        thrust::sort_by_key(thrust_par.on(stream), src_ptr, src_ptr + n,
                            dst_offset_ptr, thrust::less<uintptr_t>());
      }

      {
        const size_t block_size = options_.io_block_size;
        const size_t N = n * dim();
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        read_kernel<key_type, value_type, meta_type>
            <<<grid_size, block_size, 0, stream>>>(src, values, founds,
                                                   dst_offset, dim(), N);
      }
    }

    CudaCheckError();
  }

  /**
   * @brief Searches the hash table for the specified keys and returns address
   * of the values.
   *
   * @note When a key is missing, the data in @p values won't change.
   * @warning This API returns internal addresses for high-performance but
   * thread-unsafe. The caller is responsible for guaranteeing data consistency.
   *
   * @param n The number of key-value-meta tuples to search.
   * @param keys The keys to search on GPU-accessible memory with shape (n).
   * @param values The addresses of values to search on GPU-accessible memory
   * with shape (n).
   * @param founds The status that indicates if the keys are found on
   * GPU-accessible memory with shape (n).
   * @param metas The metas to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p metas is `nullptr`, the meta for each key will not be returned.
   * @endparblock
   * @param stream The CUDA stream that is used to execute the operation.
   *
   */
  void find(const size_type n, const key_type* keys,  // (n)
            value_type** values,                      // (n)
            bool* founds,                             // (n)
            meta_type* metas = nullptr,               // (n)
            cudaStream_t stream = 0) const {
    if (n == 0) {
      return;
    }

    CUDA_CHECK(cudaMemsetAsync(founds, 0, n * sizeof(bool), stream));

    reader_shared_lock lock(mutex_);

    using Selector = SelectLookupPtrKernel<key_type, value_type, meta_type>;
    static thread_local int step_counter = 0;
    static thread_local float load_factor = 0.0;

    if (((step_counter++) % kernel_select_interval_) == 0) {
      load_factor = fast_load_factor(0, stream, false);
    }
    Selector::execute_kernel(load_factor, options_.block_size,
                             options_.max_bucket_size, table_->buckets_num,
                             options_.dim, stream, n, d_table_, keys, values,
                             metas, founds);

    CudaCheckError();
  }

  /**
   * @brief Removes specified elements from the hash table.
   *
   * @param n The number of keys to remove.
   * @param keys The keys to remove on GPU-accessible memory.
   * @param stream The CUDA stream that is used to execute the operation.
   *
   */
  void erase(const size_type n, const key_type* keys, cudaStream_t stream = 0) {
    if (n == 0) {
      return;
    }

    write_read_lock lock(mutex_);

    {
      const size_t block_size = options_.block_size;
      const size_t N = n * TILE_SIZE;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      remove_kernel<key_type, value_type, meta_type, TILE_SIZE>
          <<<grid_size, block_size, 0, stream>>>(
              table_, keys, table_->buckets, table_->buckets_size,
              table_->bucket_max_size, table_->buckets_num, N);
    }

    CudaCheckError();
    return;
  }

  /**
   * @brief Erases all elements that satisfy the predicate @p pred from the
   * hash table.
   *
   * The value for @p pred should be a function with type `Pred` defined like
   * the following example:
   *
   *    ```
   *    template <class K, class M>
   *    __forceinline__ __device__ bool erase_if_pred(const K& key,
   *                                                  const M& meta,
   *                                                  const K& pattern,
   *                                                  const M& threshold) {
   *      return ((key & 0x1 == pattern) && (meta < threshold));
   *    }
   *    ```
   *
   * @param pred The predicate function with type Pred that returns `true` if
   * the element should be erased.
   * @param pattern The third user-defined argument to @p pred with key_type
   * type.
   * @param threshold The fourth user-defined argument to @p pred with meta_type
   * type.
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @return The number of elements removed.
   *
   */
  size_type erase_if(const Pred& pred, const key_type& pattern,
                     const meta_type& threshold, cudaStream_t stream = 0) {
    write_read_lock lock(mutex_);

    auto dev_ws{dev_mem_pool_->get_workspace<1>(sizeof(size_type), stream)};
    auto d_count{dev_ws.get<size_type*>(0)};

    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(size_type), stream));

    Pred h_pred;
    CUDA_CHECK(cudaMemcpyFromSymbolAsync(&h_pred, pred, sizeof(Pred), 0,
                                         cudaMemcpyDeviceToHost, stream));

    {
      const size_t block_size = options_.block_size;
      const size_t N = table_->buckets_num;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      remove_kernel<key_type, value_type, meta_type>
          <<<grid_size, block_size, 0, stream>>>(
              table_, h_pred, pattern, threshold, d_count, table_->buckets,
              table_->buckets_size, table_->bucket_max_size,
              table_->buckets_num, N);
    }

    size_type count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&count, d_count, sizeof(size_type),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CudaCheckError();
    return count;
  }

  /**
   * @brief Removes all of the elements in the hash table with no release
   * object.
   */
  void clear(cudaStream_t stream = 0) {
    write_read_lock lock(mutex_);

    const size_t block_size = options_.block_size;
    const size_t N = table_->buckets_num * table_->bucket_max_size;
    const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

    clear_kernel<key_type, value_type, meta_type>
        <<<grid_size, block_size, 0, stream>>>(table_, N);

    CudaCheckError();
  }

 public:
  /**
   * @brief Exports a certain number of the key-value-meta tuples from the
   * hash table.
   *
   * @param n The maximum number of exported pairs.
   * @param offset The position of the key to remove.
   * @param counter Accumulates amount of successfully exported values.
   * @param keys The keys to dump from GPU-accessible memory with shape (n).
   * @param values The values to dump from GPU-accessible memory with shape
   * (n, DIM).
   * @param metas The metas to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p metas is `nullptr`, the meta for each key will not be returned.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @return The number of elements dumped.
   *
   * @throw CudaException If the key-value size is too large for GPU shared
   * memory. Reducing the value for @p n is currently required if this exception
   * occurs.
   */
  void export_batch(size_type n, const size_type offset,
                    size_type* counter,          // (1)
                    key_type* keys,              // (n)
                    value_type* values,          // (n, DIM)
                    meta_type* metas = nullptr,  // (n)
                    cudaStream_t stream = 0) const {
    reader_shared_lock lock(mutex_);

    if (offset >= table_->capacity) {
      return;
    }
    n = std::min(table_->capacity - offset, n);

    size_type shared_size;
    size_type block_size;
    std::tie(shared_size, block_size) =
        dump_kernel_shared_memory_size<K, V, M>(shared_mem_size_);

    const size_t grid_size = SAFE_GET_GRID_SIZE(n, block_size);

    dump_kernel<key_type, value_type, meta_type>
        <<<grid_size, block_size, shared_size, stream>>>(
            table_, keys, values, metas, offset, n, counter);

    CudaCheckError();
  }

  size_type export_batch(const size_type n, const size_type offset,
                         key_type* keys,              // (n)
                         value_type* values,          // (n, DIM)
                         meta_type* metas = nullptr,  // (n)
                         cudaStream_t stream = 0) const {
    auto dev_ws{dev_mem_pool_->get_workspace<1>(sizeof(size_type), stream)};
    auto d_counter{dev_ws.get<size_type*>(0)};

    CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(size_type), stream));
    export_batch(n, offset, d_counter, keys, values, metas, stream);

    size_type counter = 0;
    CUDA_CHECK(cudaMemcpyAsync(&counter, d_counter, sizeof(size_type),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return counter;
  }

  /**
   * @brief Exports a certain number of the key-value-meta tuples which match
   * specified condition from the hash table.
   *
   * @param n The maximum number of exported pairs.
   * The value for @p pred should be a function with type `Pred` defined like
   * the following example:
   *
   *    ```
   *    template <class K, class M>
   *    __forceinline__ __device__ bool export_if_pred(const K& key,
   *                                                   M& meta,
   *                                                   const K& pattern,
   *                                                   const M& threshold) {
   *
   *      return meta > threshold;
   *    }
   *    ```
   *
   * @param pred The predicate function with type Pred that returns `true` if
   * the element should be exported.
   * @param pattern The third user-defined argument to @p pred with key_type
   * type.
   * @param threshold The fourth user-defined argument to @p pred with meta_type
   * type.
   * @param offset The position of the key to remove.
   * @param keys The keys to dump from GPU-accessible memory with shape (n).
   * @param values The values to dump from GPU-accessible memory with shape
   * (n, DIM).
   * @param metas The metas to search on GPU-accessible memory with shape (n).
   * @parblock
   * If @p metas is `nullptr`, the meta for each key will not be returned.
   * @endparblock
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @return The number of elements dumped.
   *
   * @throw CudaException If the key-value size is too large for GPU shared
   * memory. Reducing the value for @p n is currently required if this exception
   * occurs.
   */
  template <template <typename, typename> class PredFunctor>
  void export_batch_if(const key_type& pattern,
                       const meta_type& threshold, size_type n,
                       const size_type offset, size_type* d_counter,
                       key_type* keys,              // (n)
                       value_type* values,          // (n, DIM)
                       meta_type* metas = nullptr,  // (n)
                       cudaStream_t stream = 0) const {
    reader_shared_lock lock(mutex_);

    if (offset >= table_->capacity) {
      CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(size_type), stream));
      return;
    }
    n = std::min(table_->capacity - offset, n);

    const size_t meta_size = metas ? sizeof(meta_type) : 0;
    const size_t kvm_size =
        sizeof(key_type) + sizeof(value_type) * dim() + meta_size;
    const size_t block_size = std::min(shared_mem_size_ / 2 / kvm_size, 1024UL);
    MERLIN_CHECK(
        block_size > 0,
        "[HierarchicalKV] block_size <= 0, the K-V-M size may be too large!");

    const size_t shared_size = kvm_size * block_size;
    const size_t grid_size = SAFE_GET_GRID_SIZE(n, block_size);

    dump_kernel<key_type, value_type, meta_type, PredFunctor>
        <<<grid_size, block_size, shared_size, stream>>>(
            table_, pattern, threshold, keys, values, metas, offset, n,
            d_counter);

    CudaCheckError();
  }

 public:
  /**
   * @brief Indicates if the hash table has no elements.
   *
   * @param stream The CUDA stream that is used to execute the operation.
   * @return `true` if the table is empty and `false` otherwise.
   */
  bool empty(cudaStream_t stream = 0) const { return size(stream) == 0; }

  /**
   * @brief Returns the hash table size.
   *
   * @param stream The CUDA stream that is used to execute the operation.
   * @return The table size.
   */
  size_type size(cudaStream_t stream = 0) const {
    reader_shared_lock lock(mutex_);

    size_type h_size = 0;

    const size_type N = table_->buckets_num;
    const size_type step = static_cast<size_type>(
        std::numeric_limits<int>::max() / options_.max_bucket_size);

    thrust::device_ptr<int> size_ptr(table_->buckets_size);

    for (size_type start_i = 0; start_i < N; start_i += step) {
      size_type end_i = std::min(start_i + step, N);
      h_size += thrust::reduce(thrust_par.on(stream), size_ptr + start_i,
                               size_ptr + end_i, 0, thrust::plus<int>());
    }

    CudaCheckError();
    return h_size;
  }

  /**
   * @brief Returns the hash table capacity.
   *
   * @note The value that is returned might be less than the actual capacity of
   * the hash table because the hash table currently keeps the capacity to be
   * a power of 2 for performance considerations.
   *
   * @return The table capacity.
   */
  size_type capacity() const { return table_->capacity; }

  /**
   * @brief Sets the number of buckets to the number that is needed to
   * accommodate at least @p new_capacity elements without exceeding the maximum
   * load factor. This method rehashes the hash table. Rehashing puts the
   * elements into the appropriate buckets considering that total number of
   * buckets has changed.
   *
   * @note If the value of @p new_capacity or double of @p new_capacity is
   * greater or equal than `options_.max_capacity`, the reserve does not perform
   * any change to the hash table.
   *
   * @param new_capacity The requested capacity for the hash table.
   * @param stream The CUDA stream that is used to execute the operation.
   */
  void reserve(const size_type new_capacity, cudaStream_t stream = 0) {
    if (reach_max_capacity_ || new_capacity > options_.max_capacity) {
      return;
    }

    {
      write_read_lock lock(mutex_);

      // Once we have exclusive access, make sure that pending GPU calls have
      // been processed.
      CUDA_CHECK(cudaDeviceSynchronize());

      while (capacity() < new_capacity &&
             capacity() * 2 <= options_.max_capacity) {
        double_capacity(&table_);
        CUDA_CHECK(cudaDeviceSynchronize());
        sync_table_configuration();

        const size_t block_size = options_.block_size;
        const size_t N = TILE_SIZE * table_->buckets_num / 2;
        const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

        rehash_kernel_for_fast_mode<key_type, value_type, meta_type, TILE_SIZE>
            <<<grid_size, block_size, 0, stream>>>(d_table_, N);
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      reach_max_capacity_ = (capacity() * 2 > options_.max_capacity);
    }
    CudaCheckError();
  }

  /**
   * @brief Returns the average number of elements per slot, that is, size()
   * divided by capacity().
   *
   * @param stream The CUDA stream that is used to execute the operation.
   *
   * @return The load factor
   */
  float load_factor(cudaStream_t stream = 0) const {
    return static_cast<float>((size(stream) * 1.0) / (capacity() * 1.0));
  }

  /**
   * @brief Set max_capacity of the table.
   *
   * @param new_max_capacity The new expecting max_capacity. It must be power
   * of 2. Otherwise it will raise an error.
   */
  void set_max_capacity(size_type new_max_capacity) {
    if (!is_power(2, new_max_capacity)) {
      throw std::invalid_argument(
          "None power-of-2 new_max_capacity is not supported.");
    }

    write_read_lock lock(mutex_);

    if (new_max_capacity < capacity()) {
      return;
    }
    if (reach_max_capacity_) {
      reach_max_capacity_ = false;
    }
    options_.max_capacity = new_max_capacity;
  }

  /**
   * @brief Returns the dimension of the vectors.
   *
   * @return The dimension of the vectors.
   */
  size_type dim() const noexcept { return options_.dim; }

  /**
   * @brief Returns The length of each bucket.
   *
   * @return The length of each bucket.
   */
  size_type max_bucket_size() const noexcept {
    return options_.max_bucket_size;
  }

  /**
   * @brief Returns the number of buckets in the table.
   *
   * @return The number of buckets in the table.
   */
  size_type bucket_count() const noexcept { return table_->buckets_num; }

  /**
   * @brief Save keys, vectors, metas in table to file or files.
   *
   * @param file A BaseKVFile object defined the file format on host filesystem.
   * @param max_workspace_size Saving is conducted in chunks. This value denotes
   * the maximum amount of temporary memory to use when dumping the table.
   * Larger values *can* lead to higher performance.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of KV pairs saved to file.
   */
  size_type save(BaseKVFile<K, V, M>* file,
                 const size_t max_workspace_size = 1L * 1024 * 1024,
                 cudaStream_t stream = 0) const {
    const size_type tuple_size{sizeof(key_type) + sizeof(meta_type) +
                               sizeof(value_type) * dim()};
    MERLIN_CHECK(max_workspace_size >= tuple_size,
                 "[HierarchicalKV] max_workspace_size is smaller than a single "
                 "`key + metadata + value` tuple! Please set a larger value!");

    size_type shared_size;
    size_type block_size;
    std::tie(shared_size, block_size) =
        dump_kernel_shared_memory_size<K, V, M>(shared_mem_size_);

    // Request exclusive access (to make sure capacity won't change anymore).
    write_read_lock lock(mutex_);

    const size_type total_size{capacity()};
    const size_type n{std::min(max_workspace_size / tuple_size, total_size)};
    const size_type grid_size{SAFE_GET_GRID_SIZE(n, block_size)};

    // Grab temporary device and host memory.
    const size_type host_ws_size{n * tuple_size};
    auto host_ws{host_mem_pool_->get_workspace<1>(host_ws_size, stream)};
    auto h_keys{host_ws.get<key_type*>(0)};
    auto h_metas{reinterpret_cast<meta_type*>(h_keys + n)};
    auto h_values{reinterpret_cast<value_type*>(h_metas + n)};

    const size_type dev_ws_size{sizeof(size_type) + host_ws_size};
    auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
    auto d_count{dev_ws.get<size_type*>(0)};
    auto d_keys{reinterpret_cast<key_type*>(d_count + 1)};
    auto d_metas{reinterpret_cast<meta_type*>(d_keys + n)};
    auto d_values{reinterpret_cast<value_type*>(d_metas + n)};

    // Step through table, dumping contents in batches.
    size_type total_count{0};
    for (size_type i{0}; i < total_size; i += n) {
      // Dump the next batch to workspace, and then write it to the file.
      CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(size_type), stream));

      dump_kernel<key_type, value_type, meta_type>
          <<<grid_size, block_size, shared_size, stream>>>(
              table_, d_keys, d_values, d_metas, i, std::min(total_size - i, n),
              d_count);

      size_type count;
      CUDA_CHECK(cudaMemcpyAsync(&count, d_count, sizeof(size_type),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      if (count == n) {
        CUDA_CHECK(cudaMemcpyAsync(h_keys, d_keys, host_ws_size,
                                   cudaMemcpyDeviceToHost, stream));
      } else {
        CUDA_CHECK(cudaMemcpyAsync(h_keys, d_keys, sizeof(key_type) * count,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_metas, d_metas, sizeof(meta_type) * count,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_values, d_values,
                                   sizeof(value_type) * dim() * count,
                                   cudaMemcpyDeviceToHost, stream));
      }

      CUDA_CHECK(cudaStreamSynchronize(stream));
      file->write(count, dim(), h_keys, h_values, h_metas);
      total_count += count;
    }

    return total_count;
  }

  /**
   * @brief Load keys, vectors, metas from file to table.
   *
   * @param file An BaseKVFile defined the file format within filesystem.
   * @param max_workspace_size Loading is conducted in chunks. This value
   * denotes the maximum size of such chunks. Larger values *can* lead to higher
   * performance.
   * @param stream The CUDA stream used to execute the operation.
   *
   * @return Number of keys loaded from file.
   */
  size_type load(BaseKVFile<K, V, M>* file,
                 const size_t max_workspace_size = 1L * 1024 * 1024,
                 cudaStream_t stream = 0) {
    const size_type tuple_size{sizeof(key_type) + sizeof(meta_type) +
                               sizeof(value_type) * dim()};
    MERLIN_CHECK(max_workspace_size >= tuple_size,
                 "[HierarchicalKV] max_workspace_size is smaller than a single "
                 "`key + metadata + value` tuple! Please set a larger value!");

    const size_type n{max_workspace_size / tuple_size};
    const size_type ws_size{n * tuple_size};

    // Grab enough host memory to hold batch data.
    auto host_ws{host_mem_pool_->get_workspace<1>(ws_size, stream)};
    auto h_keys{host_ws.get<key_type*>(0)};
    auto h_metas{reinterpret_cast<meta_type*>(h_keys + n)};
    auto h_values{reinterpret_cast<value_type*>(h_metas + n)};

    // Attempt a first read.
    size_type count{file->read(n, dim(), h_keys, h_values, h_metas)};
    if (count == 0) {
      return 0;
    }

    // Grab equal amount of device memory as temporary storage.
    auto dev_ws{dev_mem_pool_->get_workspace<1>(ws_size, stream)};
    auto d_keys{dev_ws.get<key_type*>(0)};
    auto d_metas{reinterpret_cast<meta_type*>(d_keys + n)};
    auto d_values{reinterpret_cast<value_type*>(d_metas + n)};

    size_type total_count{0};
    do {
      if (count == n) {
        CUDA_CHECK(cudaMemcpyAsync(d_keys, h_keys, ws_size,
                                   cudaMemcpyHostToDevice, stream));
      } else {
        CUDA_CHECK(cudaMemcpyAsync(d_keys, h_keys, sizeof(key_type) * count,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_metas, h_metas, sizeof(meta_type) * count,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_values, h_values,
                                   sizeof(value_type) * dim() * count,
                                   cudaMemcpyHostToDevice, stream));
      }

      insert_or_assign(count, d_keys, d_values, d_metas, stream, true);
      total_count += count;

      // Read next batch.
      CUDA_CHECK(cudaStreamSynchronize(stream));
      count = file->read(n, dim(), h_keys, h_values, h_metas);
    } while (count > 0);

    return total_count;
  }

 private:
  bool is_power(size_t base, size_t n) {
    if (base < 2) {
      throw std::invalid_argument("is_power with zero base.");
    }
    while (n > 1) {
      if (n % base != 0) {
        return false;
      }
      n /= base;
    }
    return true;
  }

 private:
  inline bool is_fast_mode() const noexcept { return table_->is_pure_hbm; }

  /**
   * @brief Returns the load factor by sampling up to 1024 buckets.
   *
   * @note For performance consideration, the returned load factor is
   * inaccurate but within an error in 1% empirically which is enough for
   * capacity control. But it's not suitable for end-users.
   *
   * @param delta A hypothetical upcoming change on table size.
   * @param stream The CUDA stream used to execute the operation.
   * @param need_lock If lock is needed.
   *
   * @return The evaluated load factor
   */
  inline float fast_load_factor(const size_type delta = 0,
                                cudaStream_t stream = 0,
                                const bool need_lock = true) const {
    reader_shared_lock lock(mutex_, std::defer_lock);
    if (need_lock) {
      lock.lock();
    }

    size_t N = std::min(table_->buckets_num, 1024UL);

    thrust::device_ptr<int> size_ptr(table_->buckets_size);

    int size = thrust::reduce(thrust_par.on(stream), size_ptr, size_ptr + N, 0,
                              thrust::plus<int>());

    CudaCheckError();
    return static_cast<float>((delta * 1.0) / (capacity() * 1.0) +
                              (size * 1.0) /
                                  (options_.max_bucket_size * N * 1.0));
  }

  inline void check_evict_strategy(const meta_type* metas) {
    if (options_.evict_strategy == EvictStrategy::kLru) {
      MERLIN_CHECK(metas == nullptr,
                   "the metas should not be specified when running on "
                   "LRU mode.");
    }

    if (options_.evict_strategy == EvictStrategy::kCustomized) {
      MERLIN_CHECK(metas != nullptr,
                   "the metas should be specified when running on "
                   "customized mode.");
    }
  }

  /**
   * @brief Synchronize the TableCore struct to replicas.
   *
   * @note For performance consideration, synchronize the TableCore struct to
   * its replicas in constant memory and device memory when it's changed.
   */
  inline void sync_table_configuration() {
    CUDA_CHECK(
        cudaMemcpy(d_table_, table_, sizeof(TableCore), cudaMemcpyDefault));
    if (c_table_index_ >= 0) {
      CUDA_CHECK(cudaMemcpyToSymbol(c_table_, table_, sizeof(TableCore),
                                    sizeof(TableCore) * c_table_index_,
                                    cudaMemcpyDefault));
    }
  }

 private:
  HashTableOptions options_;
  TableCore* table_ = nullptr;
  TableCore* d_table_ = nullptr;
  size_t shared_mem_size_ = 0;
  std::atomic_bool reach_max_capacity_{false};
  bool initialized_ = false;
  mutable group_shared_mutex mutex_;
  const unsigned int kernel_select_interval_ = 7;
  int c_table_index_ = -1;
  std::unique_ptr<DeviceMemoryPool> dev_mem_pool_;
  std::unique_ptr<HostMemoryPool> host_mem_pool_;
};

}  // namespace merlin
}  // namespace nv
