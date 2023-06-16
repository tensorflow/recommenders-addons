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

#include <stddef.h>
#include <cstdint>
#include <cuda/std/semaphore>

namespace nv {
namespace merlin {

/**
 * Shorthand for a Key-Value-Meta tuple.
 */
template <class K, class V, class M>
struct KVM {
  K key;
  V* value;
  M meta;
};

constexpr uint64_t EMPTY_KEY = UINT64_C(0xFFFFFFFFFFFFFFFF);
constexpr uint64_t RECLAIM_KEY = UINT64_C(0xFFFFFFFFFFFFFFFE);
constexpr uint64_t VACANT_KEY_MASK = UINT64_C(0xFFFFFFFFFFFFFFFE);
constexpr uint64_t LOCKED_KEY = UINT64_C(0xFFFFFFFFFFFFFFFD);
constexpr uint64_t RESERVED_KEY_MASK = UINT64_C(0xFFFFFFFFFFFFFFFC);
constexpr uint64_t MAX_META = UINT64_C(0xFFFFFFFFFFFFFFFF);
constexpr uint64_t EMPTY_META = UINT64_C(0);

#define IS_RESERVED_KEY(key) ((RESERVED_KEY_MASK & (key)) == RESERVED_KEY_MASK)
#define IS_VACANT_KEY(key) ((VACANT_KEY_MASK & (key)) == VACANT_KEY_MASK)

template <class K>
using AtomicKey = cuda::atomic<K, cuda::thread_scope_device>;

template <class M>
using AtomicMeta = cuda::atomic<M, cuda::thread_scope_device>;

template <class T>
using AtomicPos = cuda::atomic<T, cuda::thread_scope_device>;

template <class K, class V, class M>
struct Bucket {
  AtomicKey<K>* keys_;
  AtomicMeta<M>* metas_;
  V* vectors;  // Pinned memory or HBM

  /* For upsert_kernel without user specified metas
     recording the current meta, the cur_meta will
     increment by 1 when a new inserting happens. */
  AtomicMeta<M> cur_meta;

  /* min_meta and min_pos is for or upsert_kernel
     with user specified meta. They record the minimum
     meta and its pos in the bucket. */
  AtomicMeta<M> min_meta;
  AtomicPos<int> min_pos;

  __forceinline__ __device__ AtomicKey<K>* keys(int index) const {
    return keys_ + index;
  }

  __forceinline__ __device__ AtomicMeta<M>* metas(int index) const {
    return metas_ + index;
  }
};

template <cuda::thread_scope Scope, class T = int>
class Lock {
  mutable cuda::atomic<T, Scope> _lock;

 public:
  __device__ Lock() : _lock{1} {}

  template <typename CG>
  __forceinline__ __device__ void acquire(CG const& g,
                                          unsigned long long lane = 0) const {
    if (g.thread_rank() == lane) {
      T expected = 1;
      while (!_lock.compare_exchange_weak(expected, 2,
                                          cuda::std::memory_order_acquire)) {
        expected = 1;
      }
    }
    g.sync();
  }

  template <typename CG>
  __forceinline__ __device__ void release(CG const& g,
                                          unsigned long long lane = 0) const {
    g.sync();
    if (g.thread_rank() == lane) {
      _lock.store(1, cuda::std::memory_order_release);
    }
  }
};

using Mutex = Lock<cuda::thread_scope_device>;

template <class K, class V, class M>
struct Table {
  Bucket<K, V, M>* buckets;
  Mutex* locks;                 // mutex for write buckets
  int* buckets_size;            // size of each buckets.
  V** slices;                   // Handles of the HBM/ HMEM slices.
  size_t dim;                   // Dimension of the `vectors`.
  size_t bytes_per_slice;       // Size by byte of one slice.
  size_t num_of_memory_slices;  // Number of vectors memory slices.
  size_t capacity = 134217728;  // Initial capacity.
  size_t max_size =
      std::numeric_limits<uint64_t>::max();  // Up limit of the table capacity.
  size_t buckets_num;                        // Number of the buckets.
  size_t bucket_max_size = 128;              // Volume of each buckets.
  size_t max_hbm_for_vectors = 0;            // Max HBM allocated for vectors
  size_t remaining_hbm_for_vectors = 0;  // Remaining HBM allocated for vectors
  bool is_pure_hbm = true;               // unused
  bool primary = true;                   // unused
  int slots_offset = 0;                  // unused
  int slots_number = 0;                  // unused
  int device_id = 0;                     // Device id
  int tile_size;
};

template <class K, class M>
using EraseIfPredictInternal =
    bool (*)(const K& key,       ///< iterated key in table
             M& meta,            ///< iterated meta in table
             const K& pattern,   ///< input key from caller
             const M& threshold  ///< input meta from caller
    );

/**
 * An abstract class provides interface between the nv::merlin::HashTable
 * and a file, which enables the table to save to the file or load from
 * the file, by overriding the `read` and `write` method.
 *
 * @tparam K The data type of the key.
 * @tparam V The data type of the vector's elements.
 *         The item data type should be a basic data type of C++/CUDA.
 * @tparam M The data type for `meta`.
 *           The currently supported data type is only `uint64_t`.
 *
 */
template <class K, class V, class M>
class BaseKVFile {
 public:
  virtual ~BaseKVFile() {}

  /**
   * Read from file and fill into the keys, values, and metas buffer.
   * When calling save/load method from table, it can assume that the
   * received buffer of keys, vectors, and metas are automatically
   * pre-allocated.
   *
   * @param n The number of KV pairs expect to read. `int64_t` was used
   *          here to adapt to various filesytem and formats.
   * @param dim The dimension of the `vectors`.
   * @param keys The pointer to received buffer for keys.
   * @param vectors The pointer to received buffer for vectors.
   * @param metas The pointer to received buffer for metas.
   *
   * @return Number of KV pairs have been successfully read.
   */
  virtual size_t read(const size_t n, const size_t dim, K* keys, V* vectors,
                      M* metas) = 0;

  /**
   * Write keys, values, metas from table to the file. It defines
   * an abstract method to get batch of KV pairs and write them into
   * file.
   *
   * @param n The number of KV pairs to be written. `int64_t` was used
   *          here to adapt to various filesytem and formats.
   * @param dim The dimension of the `vectors`.
   * @param keys The keys will be written to file.
   * @param vectors The vectors of values will be written to file.
   * @param metas The metas will be written to file.
   *
   * @return Number of KV pairs have been successfully written.
   */
  virtual size_t write(const size_t n, const size_t dim, const K* keys,
                       const V* vectors, const M* metas) = 0;
};

enum class OccupyResult {
  INITIAL,         ///< Initial status
  CONTINUE,        ///< Insert did not succeed, continue trying to insert
  OCCUPIED_EMPTY,  ///< New pair inserted successfully
  OCCUPIED_RECLAIMED,
  DUPLICATE,  ///< Insert did not succeed, key is already present
  EVICT,      ///< Insert succeeded by evicting one key with minimum meta.
  REFUSED,    ///< Insert did not succeed, insert meta is too low.
};

enum class OverrideResult {
  INITIAL,   ///< Initial status
  CONTINUE,  ///< Override did not succeed, continue trying to override
  SUCCESS,   ///< Override successfully
  REFUSED,   ///< Override is refused.
};

}  // namespace merlin
}  // namespace nv
