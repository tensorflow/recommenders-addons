/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http:///www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/barrier>
#include <mutex>
#include <thread>
#include <vector>
#include "types.cuh"
#include "utils.cuh"

using namespace cooperative_groups;
namespace cg = cooperative_groups;

namespace nv {
namespace merlin {

/* For improving performance consideration, allocating up to 64 table structures
 * in constant memory is supported. To close this function, please set
 * `TableOption::use_constant_memory` to `false`.
 */
constexpr int MAX_CONSTANT_TABLE = 64;
static std::mutex constant_table_mutex;
static uint64_t constant_table_flag = 0;

__constant__ char
    c_table_[sizeof(Table<uint64_t, float, uint64_t>) * MAX_CONSTANT_TABLE];

template <class T = uint64_t>
int allocate_constant_table() {
  std::lock_guard<std::mutex> guard(constant_table_mutex);
  if (constant_table_flag == std::numeric_limits<uint64_t>::max()) return -1;
  int table_index = 0;
  while (constant_table_flag & (1l << table_index)) {
    table_index++;
  }

  constant_table_flag = constant_table_flag | (1l << table_index);

  return table_index;
}

template <class T = uint64_t>
void release_constant_table(int table_index) {
  std::lock_guard<std::mutex> guard(constant_table_mutex);
  if (table_index < 0 || table_index >= MAX_CONSTANT_TABLE) return;
  constant_table_flag = constant_table_flag & (~(1l << table_index));
}

template <class M>
__global__ void create_locks(M* __restrict mutex, const size_t start,
                             const size_t end) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (start + tid < end) {
    new (mutex + start + tid) M();
  }
}

template <class M>
__global__ void release_locks(M* __restrict mutex, const size_t start,
                              const size_t end) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (start + tid < end) {
    (mutex + start + tid)->~M();
  }
}

template <class K, class V, class M>
__global__ void create_atomic_keys(Bucket<K, V, M>* __restrict buckets,
                                   const size_t start, const size_t end,
                                   const size_t bucket_max_size) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (start + tid < end) {
    for (size_t i = 0; i < bucket_max_size; i++)
      new (buckets[start + tid].keys(i))
          AtomicKey<K>{static_cast<K>(EMPTY_KEY)};
  }
}

template <class K, class V, class M>
__global__ void create_atomic_metas(Bucket<K, V, M>* __restrict buckets,
                                    const size_t start, const size_t end,
                                    const size_t bucket_max_size) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (start + tid < end) {
    for (size_t i = 0; i < bucket_max_size; i++) {
      new (buckets[start + tid].metas(i))
          AtomicMeta<M>{static_cast<M>(EMPTY_META)};
    }
    new (&(buckets[start + tid].cur_meta))
        AtomicMeta<M>{static_cast<M>(EMPTY_META)};
    new (&(buckets[start + tid].min_meta))
        AtomicMeta<M>{static_cast<M>(EMPTY_META)};
    new (&(buckets[start + tid].min_pos)) AtomicPos<int>{1};
  }
}

/* Initialize the buckets with index from start to end. */
template <class K, class V, class M>
void initialize_buckets(Table<K, V, M>** table, const size_t start,
                        const size_t end) {
  /* As testing results show us, when the number of buckets is greater than
   * the 4 million the performance will drop significantly, we believe the
   * to many pinned memory allocation causes this issue, so we change the
   * strategy to allocate some memory slices whose size is not greater than
   * 64GB, and put the buckets pointer point to the slices.
   */
  MERLIN_CHECK(start < end,
               "initialize_buckets, start should be less than end!");
  size_t buckets_num = end - start;
  const size_t total_size_of_vectors =
      buckets_num * (*table)->bucket_max_size * sizeof(V) * (*table)->dim;
  const size_t num_of_memory_slices =
      1 + (total_size_of_vectors - 1) / (*table)->bytes_per_slice;
  size_t num_of_buckets_in_one_slice =
      (*table)->bytes_per_slice /
      ((*table)->bucket_max_size * sizeof(V) * (*table)->dim);
  size_t num_of_allocated_buckets = 0;

  realloc_managed<V**>(
      &((*table)->slices), (*table)->num_of_memory_slices * sizeof(V*),
      ((*table)->num_of_memory_slices + num_of_memory_slices) * sizeof(V*));

  for (size_t i = (*table)->num_of_memory_slices;
       i < (*table)->num_of_memory_slices + num_of_memory_slices; i++) {
    if (i == (*table)->num_of_memory_slices + num_of_memory_slices - 1) {
      num_of_buckets_in_one_slice = buckets_num - num_of_allocated_buckets;
    }
    size_t slice_real_size = num_of_buckets_in_one_slice *
                             (*table)->bucket_max_size * sizeof(V) *
                             (*table)->dim;
    if ((*table)->remaining_hbm_for_vectors >= slice_real_size) {
      CUDA_CHECK(cudaMalloc(&((*table)->slices[i]), slice_real_size));
      (*table)->remaining_hbm_for_vectors -= slice_real_size;
    } else {
      (*table)->is_pure_hbm = false;
      CUDA_CHECK(
          cudaMallocHost(&((*table)->slices[i]), slice_real_size,
                         cudaHostAllocMapped | cudaHostAllocWriteCombined));
    }
    for (int j = 0; j < num_of_buckets_in_one_slice; j++) {
      (*table)->buckets[start + num_of_allocated_buckets + j].vectors =
          (*table)->slices[i] + j * (*table)->bucket_max_size * (*table)->dim;
    }
    num_of_allocated_buckets += num_of_buckets_in_one_slice;
  }

  (*table)->num_of_memory_slices += num_of_memory_slices;
  for (int i = start; i < end; i++) {
    CUDA_CHECK(cudaMalloc(&((*table)->buckets[i].keys_),
                          (*table)->bucket_max_size * sizeof(AtomicKey<K>)));
    CUDA_CHECK(cudaMalloc(&((*table)->buckets[i].metas_),
                          (*table)->bucket_max_size * sizeof(AtomicMeta<M>)));
  }

  {
    const size_t block_size = 512;
    const size_t N = end - start + 1;
    const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);
    create_locks<Mutex><<<grid_size, block_size>>>((*table)->locks, start, end);
  }

  {
    const size_t block_size = 512;
    const size_t N = end - start + 1;
    const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);
    create_atomic_keys<K, V, M><<<grid_size, block_size>>>(
        (*table)->buckets, start, end, (*table)->bucket_max_size);
  }

  {
    const size_t block_size = 512;
    const size_t N = end - start + 1;
    const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);
    create_atomic_metas<K, V, M><<<grid_size, block_size>>>(
        (*table)->buckets, start, end, (*table)->bucket_max_size);
  }
  CudaCheckError();
}

template <class K, class V, class M>
size_t get_slice_size(Table<K, V, M>** table) {
  const size_t min_slice_size =
      (*table)->bucket_max_size * sizeof(V) * (*table)->dim;
  const size_t max_table_size = (*table)->max_size * sizeof(V) * (*table)->dim;
  size_t slice_size = 0;

  if (max_table_size >= GB(16)) {
    slice_size = GB(2);
  } else if (max_table_size >= GB(2)) {
    slice_size = MB(128);
  } else if (max_table_size >= MB(128)) {
    slice_size = MB(16);
  } else if (max_table_size >= MB(16)) {
    slice_size = MB(1);
  } else {
    slice_size = min_slice_size;
  }

  return std::max(min_slice_size, slice_size);
}

/* Initialize a Table struct.

   K: The key type
   V: The value type which should be static array type and C++ class
      with customized construct is not supported.
   M: The meta type, the meta will be used to store the timestamp
      or occurrence frequency or any thing for eviction.
   DIM: Vector dimension.
*/
template <class K, class V, class M>
void create_table(Table<K, V, M>** table, const size_t dim,
                  const size_t init_size = 134217728,
                  const size_t max_size = std::numeric_limits<size_t>::max(),
                  const size_t max_hbm_for_vectors = 0,
                  const size_t bucket_max_size = 128,
                  const size_t tile_size = 32, const bool primary = true) {
  CUDA_CHECK(cudaMallocManaged((void**)table, sizeof(Table<K, V, M>)));
  CUDA_CHECK(cudaMemset(*table, 0, sizeof(Table<K, V, M>)));
  (*table)->dim = dim;
  (*table)->bucket_max_size = bucket_max_size;
  (*table)->max_size = std::max(init_size, max_size);
  (*table)->tile_size = tile_size;
  (*table)->is_pure_hbm = true;
  (*table)->bytes_per_slice = get_slice_size<K, V, M>(table);

  // The bucket number will be the minimum needed for saving memory if no
  // rehash.
  if ((init_size * 2) > (*table)->max_size) {
    (*table)->buckets_num =
        1 + (((*table)->max_size - 1) / (*table)->bucket_max_size);
  } else {
    (*table)->buckets_num = 1;
    while ((*table)->buckets_num * (*table)->bucket_max_size < init_size) {
      (*table)->buckets_num *= 2;
    }
  }

  (*table)->capacity = (*table)->buckets_num * (*table)->bucket_max_size;
  (*table)->max_hbm_for_vectors = max_hbm_for_vectors;
  (*table)->remaining_hbm_for_vectors = max_hbm_for_vectors;
  (*table)->primary = primary;

  CUDA_CHECK(cudaMalloc((void**)&((*table)->locks),
                        (*table)->buckets_num * sizeof(Mutex)));
  CUDA_CHECK(
      cudaMemset((*table)->locks, 0, (*table)->buckets_num * sizeof(Mutex)));

  CUDA_CHECK(cudaMalloc((void**)&((*table)->buckets_size),
                        (*table)->buckets_num * sizeof(int)));
  CUDA_CHECK(cudaMemset((*table)->buckets_size, 0,
                        (*table)->buckets_num * sizeof(int)));

  CUDA_CHECK(
      cudaMallocManaged((void**)&((*table)->buckets),
                        (*table)->buckets_num * sizeof(Bucket<K, V, M>)));
  CUDA_CHECK(cudaMemset((*table)->buckets, 0,
                        (*table)->buckets_num * sizeof(Bucket<K, V, M>)));

  initialize_buckets<K, V, M>(table, 0, (*table)->buckets_num);
  CudaCheckError();
}

/* Double the capacity on storage, must be followed by calling the
 * rehash_kernel. */
template <class K, class V, class M>
void double_capacity(Table<K, V, M>** table) {
  realloc<Mutex*>(&((*table)->locks), (*table)->buckets_num * sizeof(Mutex),
                  (*table)->buckets_num * sizeof(Mutex) * 2);
  realloc<int*>(&((*table)->buckets_size), (*table)->buckets_num * sizeof(int),
                (*table)->buckets_num * sizeof(int) * 2);

  realloc_managed<Bucket<K, V, M>*>(
      &((*table)->buckets), (*table)->buckets_num * sizeof(Bucket<K, V, M>),
      (*table)->buckets_num * sizeof(Bucket<K, V, M>) * 2);

  initialize_buckets<K, V, M>(table, (*table)->buckets_num,
                              (*table)->buckets_num * 2);

  (*table)->capacity *= 2;
  (*table)->buckets_num *= 2;
}

/* free all of the resource of a Table. */
template <class K, class V, class M>
void destroy_table(Table<K, V, M>** table) {
  for (int i = 0; i < (*table)->buckets_num; i++) {
    CUDA_CHECK(cudaFree((*table)->buckets[i].keys_));
    CUDA_CHECK(cudaFree((*table)->buckets[i].metas_));
  }

  for (int i = 0; i < (*table)->num_of_memory_slices; i++) {
    if (is_on_device((*table)->slices[i])) {
      CUDA_CHECK(cudaFree((*table)->slices[i]));
    } else {
      CUDA_CHECK(cudaFreeHost((*table)->slices[i]));
    }
  }
  {
    const size_t block_size = 512;
    const size_t N = (*table)->buckets_num;
    const int grid_size = SAFE_GET_GRID_SIZE(N, block_size);
    release_locks<Mutex>
        <<<grid_size, block_size>>>((*table)->locks, 0, (*table)->buckets_num);
  }
  CUDA_CHECK(cudaFree((*table)->slices));
  CUDA_CHECK(cudaFree((*table)->buckets_size));
  CUDA_CHECK(cudaFree((*table)->buckets));
  CUDA_CHECK(cudaFree((*table)->locks));
  CUDA_CHECK(cudaFree(*table));
  CUDA_CHECK(cudaDeviceSynchronize());
  CudaCheckError();
}

template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__forceinline__ __device__ void defragmentation_for_rehash(
    Bucket<K, V, M>* __restrict bucket, uint32_t remove_pos,
    const size_t bucket_max_size, const size_t buckets_num, const size_t dim) {
  uint32_t key_idx;
  size_t global_idx = 0;
  size_t start_idx = 0;
  K find_key;
  K hashed_key;

  uint32_t empty_pos = remove_pos;

  int i = 1;
  while (i < bucket_max_size) {
    key_idx = (remove_pos + i) & (bucket_max_size - 1);
    find_key = (bucket->keys(key_idx))->load(cuda::std::memory_order_relaxed);
    if (find_key == static_cast<K>(EMPTY_KEY)) {
      break;
    }
    hashed_key = Murmur3HashDevice(find_key);
    global_idx = hashed_key % (buckets_num * bucket_max_size);
    start_idx = global_idx % bucket_max_size;

    if ((start_idx <= empty_pos && empty_pos < key_idx) ||
        (key_idx < start_idx && start_idx <= empty_pos) ||
        (empty_pos <= key_idx && key_idx < start_idx)) {
      const K key =
          (*(bucket->keys(key_idx))).load(cuda::std::memory_order_relaxed);
      (*(bucket->keys(empty_pos))).store(key, cuda::std::memory_order_relaxed);
      const M meta =
          (*(bucket->metas(key_idx))).load(cuda::std::memory_order_relaxed);
      (*(bucket->metas(empty_pos)))
          .store(meta, cuda::std::memory_order_relaxed);
      for (int j = 0; j < dim; j++) {
        bucket->vectors[empty_pos * dim + j] =
            bucket->vectors[key_idx * dim + j];
      }
      (*(bucket->keys(key_idx)))
          .store(static_cast<K>(EMPTY_KEY), cuda::std::memory_order_relaxed);
      empty_pos = key_idx;
      remove_pos = key_idx;
      i = 1;
    } else {
      i++;
    }
  }
}

template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__forceinline__ __device__ void refresh_bucket_meta(
    cg::thread_block_tile<TILE_SIZE> g, Bucket<K, V, M>* bucket,
    const size_t bucket_max_size) {
  M min_val = MAX_META;
  int min_pos = 0;

  for (int i = g.thread_rank(); i < bucket_max_size; i += TILE_SIZE) {
    const K key = (bucket->keys(i))->load(cuda::std::memory_order_relaxed);
    if (key == static_cast<K>(EMPTY_KEY) ||
        key == static_cast<K>(RECLAIM_KEY)) {
      continue;
    }
    const M meta = bucket->metas(i)->load(cuda::std::memory_order_relaxed);
    if (meta < min_val) {
      min_pos = i;
      min_val = meta;
    }
  }
  M global_min_val = cg::reduce(g, min_val, cg::less<M>());
  if (min_val == global_min_val) {
    bucket->min_pos.store(min_pos, cuda::std::memory_order_relaxed);
    bucket->min_meta.store(min_val, cuda::std::memory_order_relaxed);
  }
}

template <class V, uint32_t TILE_SIZE = 4>
__device__ __forceinline__ void copy_vector(
    cg::thread_block_tile<TILE_SIZE> const& g, const V* src, V* dst,
    const size_t dim) {
  for (auto i = g.thread_rank(); i < dim; i += g.size()) {
    dst[i] = src[i];
  }

  //  cuda::barrier<cuda::thread_scope_device> bar;
  //  init(&bar, 1);
  //  cuda::memcpy_async(g, dst, src, dim * sizeof(V), bar);
  //
  //  bar.arrive_and_wait();
}

/* Write the N data from src to each address in *dst by using CPU threads,
 * usually called by upsert kernel.
 *
 * @note: In some machines with AMD CPUs, the `write_kernel` has low performance
 * thru PCI-E, so we try to use the `memcpy` on CPU threads for writing work to
 * reach better performance.
 */
template <class V>
void write_by_cpu(V** __restrict dst, const V* __restrict src,
                  const int* __restrict offset, size_t dim, int N,
                  int n_worker = 16) {
  std::vector<std::thread> thds;
  if (n_worker < 1) n_worker = 1;

  auto functor = [dim](V** __restrict dst, const V* __restrict src,
                       const int* __restrict offset, int handled_size,
                       int trunk_size) -> void {
    for (int i = handled_size; i < handled_size + trunk_size; i++) {
      if (dst[i] != nullptr) {
        memcpy(dst[i], src + offset[i] * dim, sizeof(V) * dim);
      }
    }
  };

  int32_t trunk_size_floor = N / n_worker;
  int32_t trunk_size_remain = N % n_worker;
  int32_t n_worker_used = trunk_size_floor == 0 ? trunk_size_remain : n_worker;

  size_t handled_size = 0;
  for (int i = 0; i < n_worker_used; i++) {
    int32_t cur_trunk_size = trunk_size_floor;
    if (trunk_size_remain != 0) {
      cur_trunk_size += 1;
      trunk_size_remain--;
    }
    thds.push_back(
        std::thread(functor, dst, src, offset, handled_size, cur_trunk_size));
    handled_size += cur_trunk_size;
  }

  for (int i = 0; i < n_worker_used; i++) {
    thds[i].join();
  }
}

template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__forceinline__ __device__ void move_key_to_new_bucket(
    cg::thread_block_tile<TILE_SIZE> g, int rank, const K& key, const M& meta,
    const V* __restrict vector, Bucket<K, V, M>* __restrict new_bucket,
    const size_t new_bkt_idx, const size_t new_start_idx,
    int* __restrict buckets_size, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim) {
  uint32_t key_pos;
  unsigned empty_vote;
  int local_size;
  int src_lane;

  for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    size_t key_offset =
        (new_start_idx + tile_offset + rank) & (bucket_max_size - 1);
    const K current_key =
        (*(new_bucket->keys(key_offset))).load(cuda::std::memory_order_relaxed);
    empty_vote = g.ballot(current_key == static_cast<K>(EMPTY_KEY));
    if (empty_vote) {
      src_lane = __ffs(empty_vote) - 1;
      key_pos =
          (new_start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
      local_size = buckets_size[new_bkt_idx];
      if (rank == src_lane) {
        new_bucket->keys(key_pos)->store(key, cuda::std::memory_order_relaxed);
        new_bucket->metas(key_pos)->store(meta,
                                          cuda::std::memory_order_relaxed);
        atomicAdd(&(buckets_size[new_bkt_idx]), 1);
      }
      local_size = g.shfl(local_size, src_lane);
      if (local_size >= bucket_max_size) {
        refresh_bucket_meta<K, V, M, TILE_SIZE>(g, new_bucket, bucket_max_size);
      }
      copy_vector<V, TILE_SIZE>(g, vector, new_bucket->vectors + key_pos * dim,
                                dim);
      break;
    }
  }
}

template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__global__ void rehash_kernel_for_fast_mode(
    const Table<K, V, M>* __restrict table, size_t N) {
  Bucket<K, V, M>* buckets = table->buckets;
  int* __restrict buckets_size = table->buckets_size;
  const size_t bucket_max_size = table->bucket_max_size;
  const size_t buckets_num = table->buckets_num;
  const size_t dim = table->dim;

  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t global_idx;
  uint32_t start_idx = 0;
  K target_key = 0;
  M target_meta = 0;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    uint32_t bkt_idx = t / TILE_SIZE;
    Bucket<K, V, M>* bucket = (buckets + bkt_idx);

    lock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
    uint32_t key_idx = 0;
    while (key_idx < bucket_max_size) {
      key_idx = g.shfl(key_idx, 0);
      target_key =
          (bucket->keys(key_idx))->load(cuda::std::memory_order_relaxed);
      target_meta =
          bucket->metas(key_idx)->load(cuda::std::memory_order_relaxed);
      if (target_key != static_cast<K>(EMPTY_KEY) &&
          target_key != static_cast<K>(RECLAIM_KEY)) {
        K hashed_key = Murmur3HashDevice(target_key);
        global_idx = hashed_key % (buckets_num * bucket_max_size);
        uint32_t new_bkt_idx = global_idx / bucket_max_size;
        if (new_bkt_idx != bkt_idx) {
          start_idx = global_idx % bucket_max_size;
          move_key_to_new_bucket<K, V, M, TILE_SIZE>(
              g, rank, target_key, target_meta,
              (bucket->vectors + key_idx * dim), buckets + new_bkt_idx,
              new_bkt_idx, start_idx, buckets_size, bucket_max_size,
              buckets_num, table->dim);
          if (rank == 0) {
            (bucket->keys(key_idx))
                ->store(static_cast<K>(EMPTY_KEY),
                        cuda::std::memory_order_relaxed);
            atomicSub(&(buckets_size[bkt_idx]), 1);
            defragmentation_for_rehash<K, V, M, TILE_SIZE>(
                bucket, key_idx, bucket_max_size, buckets_num / 2, dim);
            key_idx = 0;
          }
        } else {
          key_idx++;
        }
      } else {
        key_idx++;
      }
    }
    unlock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
  }
}

/* Write the N data from src to each address in *dst,
   usually called by upsert kernel.

   `src`: A continuous memory pointer with Vector
          which can be HBM.
   `dst`: A pointer of pointer to V which should be on HBM,
          but each value (a pointer of V) could point to a
          memory on HBM or HMEM.
   `N`: Number of vectors that need to be written.
*/
template <class K, class V, class M>
__global__ void write_kernel(const V* __restrict src, V** __restrict dst,
                             const int* __restrict src_offset, const size_t dim,
                             const size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / dim);
    int dim_index = t % dim;

    if (dst[vec_index] != nullptr) {
      if (src_offset != nullptr) {
        dst[vec_index][dim_index] =
            src[src_offset[vec_index] * dim + dim_index];
      } else {
        dst[vec_index][dim_index] = src[vec_index * dim + dim_index];
      }
    }
  }
}

/* Write the values of delta_or_val into the table. If the key[i] is already in
   the table indicted be @exists[i], a @delta_or_val[i] will be added to the the
   existing value. if the key not exists, the value @val_or_delta[i] will be
   assigned to the address @dst[i].

   `delta_or_val`: will be treated as val and accumlating should be executed.
   `dst`: A pointer of pointer to V which should be on HBM,
          but each value (a pointer of V) could point to a
          memory on HBM or HMEM.
   `existed`: If the keys existed before this kernel is executed.
   `status`: The existence status for each key when the kernel is being
   executed.

   `N`: number of vectors needed to be writen.
*/
template <class K, class V, class M>
__global__ void write_with_accum_kernel(const V* __restrict delta_or_val,
                                        V** __restrict dst,
                                        const bool* __restrict existed,
                                        const bool* __restrict status,
                                        const int* __restrict src_offset,
                                        const size_t dim, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / dim);
    int dim_index = t % dim;

    if (dst[vec_index] != nullptr &&
        existed[src_offset[vec_index]] == status[src_offset[vec_index]]) {
      if (status[src_offset[vec_index]]) {
        dst[vec_index][dim_index] +=
            delta_or_val[src_offset[vec_index] * dim + dim_index];
      } else {
        dst[vec_index][dim_index] =
            delta_or_val[src_offset[vec_index] * dim + dim_index];
      }
    }
  }
}

/* Add a @delta[i] to the the value saved in the address @dst[i].

   `delta`: a delta value which should be add to.
   `dst`: A pointer of pointer to V which should be on HBM,
          but each value (a pointer of V) could point to a
          memory on HBM or HMEM.
   `N`: number of vectors needed to be writen.
*/
template <class K, class V, class M>
__global__ void write_with_accum_kernel(const V* __restrict delta,
                                        V** __restrict dst,
                                        const int* __restrict src_offset,
                                        const size_t dim, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / dim);
    int dim_index = t % dim;

    if (dst[vec_index] != nullptr) {
      dst[vec_index][dim_index] +=
          delta[src_offset[vec_index] * dim + dim_index];
    }
  }
}

/* Read the N data from src to each address in *dst,
   usually called by upsert kernel.

   `src`: A pointer of pointer of V which should be on HBM,
          but each value (a pointer of V) could point to a
          memory on HBM or HMEM.
   `dst`: A continue memory pointer with Vector
          which should be HBM.
   `mask`: One for each `dst`. If true, reading from src,
           or false reading from default_val.
   `default_val`: Default value with shape (1, DIM) or (N, DIM)
   `N`: The number of vectors needed to be read.
   'full_size_default':
      If true, the d_def_val will be treated as
      a full size default value which shape must be (N, DIM).
*/
template <class K, class V, class M>
__global__ void read_kernel(const V* const* __restrict src, V* __restrict dst,
                            const bool* mask, const int* __restrict dst_offset,
                            const size_t dim, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / dim);
    int dim_index = t % dim;
    int real_dst_offset =
        dst_offset != nullptr ? dst_offset[vec_index] : vec_index;

    /// Copy selected values and fill in default value for all others.
    if (mask[real_dst_offset] && src[vec_index] != nullptr) {
      dst[real_dst_offset * dim + dim_index] = src[vec_index][dim_index];
    }
  }
}

/* Read the N data from src to each address in *dst,
 *  usually called by upsert kernel.
 *
 *  `src`: A pointer of pointer of V which should be on HBM,
 *         but each value (a pointer of V) could point to a
 *         memory on HBM or HMEM.
 *  `dst`: A continue memory pointer with Vector
 *         which should be HBM.
 *  `N`: Number of vectors needed to be read.
 */
template <class K, class V, class M>
__global__ void read_kernel(const V** __restrict src, V* __restrict dst,
                            const int* __restrict dst_offset, const size_t dim,
                            const size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / dim);
    int real_dst_offset =
        dst_offset != nullptr ? dst_offset[vec_index] : vec_index;
    int dim_index = t % dim;
    if (src[vec_index] != nullptr) {
      dst[real_dst_offset * dim + dim_index] = src[vec_index * dim + dim_index];
    }
  }
}

template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__device__ __forceinline__ unsigned find_in_bucket(
    cg::thread_block_tile<TILE_SIZE> g, Bucket<K, V, M>* bucket,
    const K& find_key, uint32_t& tile_offset, const uint32_t& start_idx,
    const size_t& bucket_max_size) {
  uint32_t key_pos = 0;

#pragma unroll
  for (tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    key_pos =
        (start_idx + tile_offset + g.thread_rank()) & (bucket_max_size - 1);
    auto const current_key =
        bucket->keys(key_pos)->load(cuda::std::memory_order_relaxed);
    auto const found_vote = g.ballot(find_key == current_key);
    if (found_vote) {
      return found_vote;
    }

    if (g.any(current_key == static_cast<K>(EMPTY_KEY))) {
      return 0;
    }
  }
  return 0;
}

template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__device__ __forceinline__ OccupyResult find_without_lock(
    cg::thread_block_tile<TILE_SIZE> g, Bucket<K, V, M>* __restrict__ bucket,
    const K desired_key, const size_t start_idx, int& key_pos, int& src_lane,
    const size_t bucket_max_size) {
  K expected_key = static_cast<K>(EMPTY_KEY);

  AtomicKey<K>* current_key;

  unsigned vote = 0;

  for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    key_pos = (start_idx + tile_offset + g.thread_rank()) % bucket_max_size;

    current_key = bucket->keys(key_pos);

    expected_key = current_key->load(cuda::std::memory_order_relaxed);
    vote = g.ballot(desired_key == expected_key);
    if (vote) {
      src_lane = __ffs(vote) - 1;
      key_pos = g.shfl(key_pos, src_lane);
      return OccupyResult::DUPLICATE;
    }
    vote = g.ballot(expected_key == static_cast<K>(EMPTY_KEY));
    if (vote) break;
  }
  return OccupyResult::CONTINUE;
}

template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__device__ __inline__ OccupyResult find_and_lock_when_vacant(
    cg::thread_block_tile<TILE_SIZE> g, Bucket<K, V, M>* __restrict__ bucket,
    const K desired_key, const M desired_meta, K& evicted_key,
    const size_t start_idx, int& key_pos, int& src_lane,
    const size_t bucket_max_size) {
  K expected_key = static_cast<K>(EMPTY_KEY);

  AtomicKey<K>* current_key;
  AtomicMeta<M>* current_meta;

  K local_min_meta_key = static_cast<K>(EMPTY_KEY);

  M local_min_meta_val = MAX_META;
  M temp_min_meta_val = MAX_META;
  int local_min_meta_pos = -1;

  unsigned vote = 0;
  bool result = false;

  for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    key_pos = (start_idx + tile_offset + g.thread_rank()) % bucket_max_size;

    current_key = bucket->keys(key_pos);

    // Step 1: try find and lock the desired_key.
    do {
      expected_key = desired_key;
      result = current_key->compare_exchange_strong(
          expected_key, static_cast<K>(LOCKED_KEY),
          cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);
      vote = g.ballot(result);
      if (vote) {
        src_lane = __ffs(vote) - 1;
        key_pos = g.shfl(key_pos, src_lane);
        return OccupyResult::DUPLICATE;
      }
      vote = g.ballot(expected_key == static_cast<K>(EMPTY_KEY));
      if (vote) break;
      vote = g.ballot(expected_key == static_cast<K>(LOCKED_KEY));
    } while (vote != 0);

    // Step 2: (TBD)try find empty location.
    while (vote) {
      src_lane = __ffs(vote) - 1;
      if (src_lane == g.thread_rank()) {
        expected_key = static_cast<K>(EMPTY_KEY);
        result = current_key->compare_exchange_strong(
            expected_key, static_cast<K>(LOCKED_KEY),
            cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);
      }
      result = g.shfl(result, src_lane);
      if (result) {
        key_pos = g.shfl(key_pos, src_lane);
        return OccupyResult::OCCUPIED_EMPTY;
      }
      vote -= ((unsigned(0x1)) << src_lane);
    }
  }

  for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    key_pos = (start_idx + tile_offset + g.thread_rank()) % bucket_max_size;

    current_meta = bucket->metas(key_pos);

    // Step 4: record min meta location.
    temp_min_meta_val = current_meta->load(cuda::std::memory_order_relaxed);
    if (temp_min_meta_val < local_min_meta_val) {
      expected_key =
          bucket->keys(key_pos)->load(cuda::std::memory_order_relaxed);
      if (expected_key != static_cast<K>(LOCKED_KEY) &&
          expected_key != static_cast<K>(EMPTY_KEY)) {
        local_min_meta_key = expected_key;
        local_min_meta_val = temp_min_meta_val;
        local_min_meta_pos = key_pos;
      }
    }
  }
  // Step 5: insert by evicting some one.
  const M global_min_meta_val =
      cg::reduce(g, local_min_meta_val, cg::less<M>());
  if (desired_meta < global_min_meta_val) {
    return OccupyResult::REFUSED;
  }
  vote = g.ballot(local_min_meta_val <= global_min_meta_val);
  if (vote) {
    src_lane = __ffs(vote) - 1;
    result = false;
    if (src_lane == g.thread_rank()) {
      // TBD: Here can be compare_exchange_weak. Do benchmark.
      current_key = bucket->keys(local_min_meta_pos);
      current_meta = bucket->metas(local_min_meta_pos);
      evicted_key = local_min_meta_key;
      result = current_key->compare_exchange_strong(
          local_min_meta_key, static_cast<K>(LOCKED_KEY),
          cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);

      // Need to recover when fail.
      if (result && (current_meta->load(cuda::std::memory_order_relaxed) >
                     global_min_meta_val)) {
        current_key->store(local_min_meta_key, cuda::std::memory_order_relaxed);
        result = false;
      }
    }
    result = g.shfl(result, src_lane);
    if (result) {
      // Not every `evicted_key` is correct expect the `src_lane` thread.
      key_pos = g.shfl(local_min_meta_pos, src_lane);
      return (evicted_key == static_cast<K>(RECLAIM_KEY))
                 ? OccupyResult::OCCUPIED_RECLAIMED
                 : OccupyResult::EVICT;
    }
  }
  return OccupyResult::CONTINUE;
}

template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__device__ __forceinline__ OccupyResult find_and_lock_when_full(
    cg::thread_block_tile<TILE_SIZE> g, Bucket<K, V, M>* __restrict__ bucket,
    const K desired_key, const M desired_meta, K& evicted_key,
    const size_t start_idx, int& key_pos, int& src_lane,
    const size_t bucket_max_size) {
  K expected_key = static_cast<K>(EMPTY_KEY);

  AtomicKey<K>* current_key;
  AtomicMeta<M>* current_meta;

  K local_min_meta_key = static_cast<K>(EMPTY_KEY);

  M local_min_meta_val = MAX_META;
  M temp_min_meta_val = MAX_META;
  int local_min_meta_pos = -1;

  unsigned vote = 0;
  bool result = false;

  for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    key_pos = (start_idx + tile_offset + g.thread_rank()) % bucket_max_size;

    current_key = bucket->keys(key_pos);

    // Step 1: try find and lock the desired_key.
    do {
      expected_key = desired_key;
      result = current_key->compare_exchange_strong(
          expected_key, static_cast<K>(LOCKED_KEY),
          cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);
      vote = g.ballot(result);
      if (vote) {
        src_lane = __ffs(vote) - 1;
        key_pos = g.shfl(key_pos, src_lane);
        return OccupyResult::DUPLICATE;
      }
      vote = g.ballot(expected_key == static_cast<K>(LOCKED_KEY));
    } while (vote != 0);
  }

  for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    key_pos = (start_idx + tile_offset + g.thread_rank()) % bucket_max_size;

    // Step 2: record min meta location.
    temp_min_meta_val =
        bucket->metas(key_pos)->load(cuda::std::memory_order_relaxed);
    if (temp_min_meta_val < local_min_meta_val) {
      while ((expected_key = bucket->keys(key_pos)->load(
                  cuda::std::memory_order_relaxed)) ==
             static_cast<K>(LOCKED_KEY))
        ;
      local_min_meta_key = expected_key;
      local_min_meta_val = temp_min_meta_val;
      local_min_meta_pos = key_pos;
    }
  }

  // Step 3: insert by evicting some one.
  const M global_min_meta_val =
      cg::reduce(g, local_min_meta_val, cg::less<M>());
  if (desired_meta < global_min_meta_val) {
    return OccupyResult::REFUSED;
  }
  vote = g.ballot(local_min_meta_val <= global_min_meta_val);
  if (vote) {
    src_lane = __ffs(vote) - 1;
    result = false;
    if (src_lane == g.thread_rank()) {
      // TBD: Here can be compare_exchange_weak. Do benchmark.
      current_key = bucket->keys(local_min_meta_pos);
      current_meta = bucket->metas(local_min_meta_pos);
      evicted_key = local_min_meta_key;
      result = current_key->compare_exchange_strong(
          local_min_meta_key, static_cast<K>(LOCKED_KEY),
          cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);

      // Need to recover when fail.
      if (result && (current_meta->load(cuda::std::memory_order_relaxed) >
                     global_min_meta_val)) {
        current_key->store(local_min_meta_key, cuda::std::memory_order_relaxed);
        result = false;
      }
    }
    result = g.shfl(result, src_lane);
    if (result) {
      // Not every `evicted_key` is correct expect the `src_lane` thread.
      key_pos = g.shfl(local_min_meta_pos, src_lane);
      return (evicted_key == static_cast<K>(RECLAIM_KEY))
                 ? OccupyResult::OCCUPIED_RECLAIMED
                 : OccupyResult::EVICT;
    }
  }
  return OccupyResult::CONTINUE;
}

template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__device__ __forceinline__ OccupyResult find_and_lock_for_update(
    cg::thread_block_tile<TILE_SIZE> g, Bucket<K, V, M>* __restrict__ bucket,
    const K desired_key, const size_t start_idx, int& key_pos, int& src_lane,
    const size_t bucket_max_size) {
  K expected_key = static_cast<K>(EMPTY_KEY);

  AtomicKey<K>* current_key;

  unsigned vote = 0;
  bool result = false;

  for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
       tile_offset += TILE_SIZE) {
    key_pos = (start_idx + tile_offset + g.thread_rank()) % bucket_max_size;

    current_key = bucket->keys(key_pos);

    // Step 1: try find and lock the desired_key.
    do {
      expected_key = desired_key;
      result = current_key->compare_exchange_strong(
          expected_key, static_cast<K>(LOCKED_KEY),
          cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed);
      vote = g.ballot(result);
      if (vote) {
        src_lane = __ffs(vote) - 1;
        key_pos = g.shfl(key_pos, src_lane);
        return OccupyResult::DUPLICATE;
      }
      vote = g.ballot(expected_key == static_cast<K>(EMPTY_KEY));
      if (vote) return OccupyResult::REFUSED;
      vote = g.ballot(expected_key == static_cast<K>(LOCKED_KEY));
    } while (vote != 0);
  }
  return OccupyResult::REFUSED;
}

template <class K, class V, class M>
__forceinline__ __device__ Bucket<K, V, M>* get_key_position(
    Bucket<K, V, M>* __restrict buckets, const K key, size_t& bkt_idx,
    size_t& start_idx, const size_t buckets_num, const size_t bucket_max_size) {
  const uint32_t hashed_key = Murmur3HashDevice(key);
  const size_t global_idx = hashed_key % (buckets_num * bucket_max_size);
  bkt_idx = global_idx / bucket_max_size;
  start_idx = global_idx % bucket_max_size;
  return buckets + bkt_idx;
}

template <class K, class V, class M>
__forceinline__ __device__ void update_meta(Bucket<K, V, M>* __restrict bucket,
                                            const int key_pos,
                                            const M* __restrict metas,
                                            const int key_idx) {
  if (metas == nullptr) {
    M cur_meta =
        bucket->cur_meta.fetch_add(1, cuda::std::memory_order_relaxed) + 1;
    bucket->metas(key_pos)->store(cur_meta, cuda::std::memory_order_relaxed);
  } else {
    bucket->metas(key_pos)->store(metas[key_idx],
                                  cuda::std::memory_order_relaxed);
  }
  return;
}

template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__global__ void upsert_kernel_with_io_core(
    const Table<K, V, M>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    const V* __restrict values, const M* __restrict metas, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;

    const K insert_key = keys[key_idx];

    if (IS_RESERVED_KEY(insert_key)) continue;

    const M insert_meta =
        metas != nullptr ? metas[key_idx] : static_cast<M>(MAX_META);
    const V* insert_value = values + key_idx * dim;

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;
    K evicted_key;

    Bucket<K, V, M>* bucket =
        get_key_position<K>(table->buckets, insert_key, bkt_idx, start_idx,
                            buckets_num, bucket_max_size);

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    do {
      if (bucket_size < bucket_max_size) {
        occupy_result = find_and_lock_when_vacant<K, V, M, TILE_SIZE>(
            g, bucket, insert_key, insert_meta, evicted_key, start_idx, key_pos,
            src_lane, bucket_max_size);
      } else {
        start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
        occupy_result = find_and_lock_when_full<K, V, M, TILE_SIZE>(
            g, bucket, insert_key, insert_meta, evicted_key, start_idx, key_pos,
            src_lane, bucket_max_size);
      }

      occupy_result = g.shfl(occupy_result, src_lane);
    } while (occupy_result == OccupyResult::CONTINUE);

    if (occupy_result == OccupyResult::REFUSED) continue;

    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        g.thread_rank() == src_lane) {
      atomicAdd(&(buckets_size[bkt_idx]), 1);
    }

    copy_vector<V, TILE_SIZE>(g, insert_value, bucket->vectors + key_pos * dim,
                              dim);
    if (g.thread_rank() == src_lane) {
      update_meta(bucket, key_pos, metas, key_idx);
      (bucket->keys(key_pos))
          ->store(insert_key, cuda::std::memory_order_relaxed);
    }
  }
}

template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__global__ void upsert_and_evict_kernel_with_io_core(
    const Table<K, V, M>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    const V* __restrict values, const M* __restrict metas,
    K* __restrict evicted_keys, V* __restrict evicted_values,
    M* __restrict evicted_metas, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    const size_t key_idx = t / TILE_SIZE;

    const K insert_key = keys[key_idx];

    if (IS_RESERVED_KEY(insert_key)) continue;

    const M insert_meta =
        metas != nullptr ? metas[key_idx] : static_cast<M>(MAX_META);
    const V* insert_value = values + key_idx * dim;

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;
    K evicted_key;

    Bucket<K, V, M>* bucket =
        get_key_position<K>(table->buckets, insert_key, bkt_idx, start_idx,
                            buckets_num, bucket_max_size);

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    do {
      if (bucket_size < bucket_max_size) {
        occupy_result = find_and_lock_when_vacant<K, V, M, TILE_SIZE>(
            g, bucket, insert_key, insert_meta, evicted_key, start_idx, key_pos,
            src_lane, bucket_max_size);
      } else {
        start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
        occupy_result = find_and_lock_when_full<K, V, M, TILE_SIZE>(
            g, bucket, insert_key, insert_meta, evicted_key, start_idx, key_pos,
            src_lane, bucket_max_size);
      }
      occupy_result = g.shfl(occupy_result, src_lane);
    } while (occupy_result == OccupyResult::CONTINUE);

    if (occupy_result == OccupyResult::REFUSED) {
      copy_vector<V, TILE_SIZE>(g, insert_value, evicted_values + key_idx * dim,
                                dim);
      continue;
    }

    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        g.thread_rank() == src_lane) {
      atomicAdd(&(buckets_size[bkt_idx]), 1);
    }

    if (occupy_result == OccupyResult::EVICT) {
      if (g.thread_rank() == src_lane) {
        evicted_keys[key_idx] = evicted_key;
      }
      if (metas != nullptr) {
        evicted_metas[key_idx] = metas[key_idx];
      }
      copy_vector<V, TILE_SIZE>(g, bucket->vectors + key_pos * dim,
                                evicted_values + key_idx * dim, dim);
    }

    copy_vector<V, TILE_SIZE>(g, insert_value, bucket->vectors + key_pos * dim,
                              dim);
    if (g.thread_rank() == src_lane) {
      update_meta(bucket, key_pos, metas, key_idx);
      (bucket->keys(key_pos))
          ->store(insert_key, cuda::std::memory_order_relaxed);
    }
  }
}

template <typename K, typename V, typename M>
struct SelectUpsertKernelWithIO {
  static void execute_kernel(const float& load_factor, const int& block_size,
                             const size_t bucket_max_size,
                             const size_t buckets_num, const size_t dim,
                             cudaStream_t& stream, const size_t& n,
                             const Table<K, V, M>* __restrict table,
                             const K* __restrict keys,
                             const V* __restrict values,
                             const M* __restrict metas) {
    if (load_factor <= 0.5) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_kernel_with_io_core<K, V, M, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, values, metas, N);

    } else if (load_factor <= 0.875) {
      const unsigned int tile_size = 8;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_kernel_with_io_core<K, V, M, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, values, metas, N);
    } else {
      const unsigned int tile_size = 32;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_kernel_with_io_core<K, V, M, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, values, metas, N);
    }
    return;
  }
};

template <typename K, typename V, typename M>
struct SelectUpsertAndEvictKernelWithIO {
  static void execute_kernel(
      const float& load_factor, const int& block_size,
      const size_t bucket_max_size, const size_t buckets_num, const size_t dim,
      cudaStream_t& stream, const size_t& n,
      const Table<K, V, M>* __restrict table, const K* __restrict keys,
      const V* __restrict values, const M* __restrict metas,
      K* __restrict evicted_keys, V* __restrict evicted_values,
      M* __restrict evicted_metas) {
    if (load_factor <= 0.5) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_and_evict_kernel_with_io_core<K, V, M, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, values, metas,
              evicted_keys, evicted_values, evicted_metas, N);

    } else if (load_factor <= 0.875) {
      const unsigned int tile_size = 8;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);

      upsert_and_evict_kernel_with_io_core<K, V, M, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, values, metas,
              evicted_keys, evicted_values, evicted_metas, N);

    } else {
      const unsigned int tile_size = 32;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      upsert_and_evict_kernel_with_io_core<K, V, M, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, values, metas,
              evicted_keys, evicted_values, evicted_metas, N);
    }
    return;
  }
};

/* Upsert with the end-user specified meta.
 */
template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__global__ void upsert_kernel(const Table<K, V, M>* __restrict table,
                              const size_t bucket_max_size,
                              const size_t buckets_num, const size_t dim,
                              const K* __restrict keys, V** __restrict vectors,
                              const M* __restrict metas,
                              int* __restrict src_offset, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;

    const K insert_key = keys[key_idx];
    if (IS_RESERVED_KEY(insert_key)) continue;

    const M insert_meta =
        metas != nullptr ? metas[key_idx] : static_cast<M>(MAX_META);

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;
    K evicted_key;

    Bucket<K, V, M>* bucket =
        get_key_position<K>(table->buckets, insert_key, bkt_idx, start_idx,
                            buckets_num, bucket_max_size);

    if (src_offset != nullptr && g.thread_rank() == 0) {
      *(src_offset + key_idx) = key_idx;
    }

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    do {
      if (bucket_size < bucket_max_size) {
        occupy_result = find_and_lock_when_vacant<K, V, M, TILE_SIZE>(
            g, bucket, insert_key, insert_meta, evicted_key, start_idx, key_pos,
            src_lane, bucket_max_size);
      } else {
        start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
        occupy_result = find_and_lock_when_vacant<K, V, M, TILE_SIZE>(
            g, bucket, insert_key, insert_meta, evicted_key, start_idx, key_pos,
            src_lane, bucket_max_size);
      }

      occupy_result = g.shfl(occupy_result, src_lane);
    } while (occupy_result == OccupyResult::CONTINUE);

    if (occupy_result == OccupyResult::REFUSED) continue;

    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        g.thread_rank() == src_lane) {
      atomicAdd(&(buckets_size[bkt_idx]), 1);
    }

    if (g.thread_rank() == src_lane) {
      *(vectors + key_idx) = (bucket->vectors + key_pos * dim);
      update_meta(bucket, key_pos, metas, key_idx);
      (bucket->keys(key_pos))
          ->store(insert_key, cuda::std::memory_order_relaxed);
    }
  }
}

/* Accum kernel with customized metas.
 */
template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__global__ void accum_kernel(
    const Table<K, V, M>* __restrict table, const K* __restrict keys,
    V** __restrict vectors, const M* __restrict metas,
    const bool* __restrict existed, Bucket<K, V, M>* __restrict buckets,
    int* __restrict buckets_size, const size_t bucket_max_size,
    const size_t buckets_num, int* __restrict src_offset,
    bool* __restrict status, size_t N) {
  const size_t dim = table->dim;
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    int local_size = 0;
    bool local_found = false;
    unsigned found_or_empty_vote = 0;

    size_t key_idx = t / TILE_SIZE;
    K insert_key = *(keys + key_idx);

    if (IS_RESERVED_KEY(insert_key)) continue;

    K hashed_key = Murmur3HashDevice(insert_key);
    size_t global_idx = hashed_key % (buckets_num * bucket_max_size);
    size_t bkt_idx = global_idx / bucket_max_size;
    size_t start_idx = global_idx % bucket_max_size;

    int src_lane = -1;

    Bucket<K, V, M>* bucket = buckets + bkt_idx;
    lock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
    if (rank == 0 && src_offset != nullptr) {
      *(src_offset + key_idx) = key_idx;
    }

    for (uint32_t tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      size_t key_offset =
          (start_idx + tile_offset + rank) & (bucket_max_size - 1);
      K current_key =
          bucket->keys(key_offset)->load(cuda::std::memory_order_relaxed);
      found_or_empty_vote = g.ballot(current_key == static_cast<K>(EMPTY_KEY) ||
                                     insert_key == current_key);
      if (found_or_empty_vote) {
        src_lane = __ffs(found_or_empty_vote) - 1;
        key_pos = (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
        local_size = buckets_size[bkt_idx];
        if (rank == src_lane) {
          if (current_key == insert_key) {
            local_found = true;
            *(status + key_idx) = local_found;
          }
          if (local_found == existed[key_idx]) {
            (bucket->keys(key_pos))
                ->store(insert_key, cuda::std::memory_order_relaxed);
            if (!local_found) {
              buckets_size[bkt_idx]++;
              local_size++;
            }
            *(vectors + key_idx) = (bucket->vectors + key_pos * dim);
            update_meta(bucket, key_pos, metas, key_idx);
          }
        }
        local_size = g.shfl(local_size, src_lane);
        if (local_size >= bucket_max_size) {
          refresh_bucket_meta<K, V, M, TILE_SIZE>(g, bucket, bucket_max_size);
        }
        break;
      }
    }
    if (!found_or_empty_vote) {
      if (rank == (bucket->min_pos % TILE_SIZE)) {
        key_pos = bucket->min_pos;
        (bucket->keys(key_pos))
            ->store(insert_key, cuda::std::memory_order_relaxed);
        *(vectors + key_idx) = (bucket->vectors + key_pos * dim);
        update_meta(bucket, key_pos, metas, key_idx);
      }
      refresh_bucket_meta<K, V, M, TILE_SIZE>(g, bucket, bucket_max_size);
    }
    unlock<Mutex, TILE_SIZE>(g, table->locks[bkt_idx]);
  }
}

/* lookup with IO operation. This kernel is
 * usually used for the pure HBM mode for better performance.
 */
template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__global__ void lookup_kernel_with_io(const Table<K, V, M>* __restrict table,
                                      const size_t bucket_max_size,
                                      const size_t buckets_num,
                                      const size_t dim,
                                      const K* __restrict keys,
                                      V* __restrict values, M* __restrict metas,
                                      bool* __restrict found, size_t N) {
  int* buckets_size = table->buckets_size;
  Bucket<K, V, M>* buckets = table->buckets;

  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_idx = t / TILE_SIZE;

    const K find_key = keys[key_idx];
    if (IS_RESERVED_KEY(find_key)) continue;

    V* find_value = values + key_idx * dim;

    int key_pos = -1;
    int src_lane = -1;
    size_t bkt_idx = 0;
    size_t start_idx = 0;

    Bucket<K, V, M>* bucket = get_key_position<K>(
        buckets, find_key, bkt_idx, start_idx, buckets_num, bucket_max_size);

    const int bucket_size = buckets_size[bkt_idx];
    if (bucket_size >= bucket_max_size) {
      start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
    }

    OccupyResult occupy_result{OccupyResult::INITIAL};
    occupy_result = find_without_lock<K, V, M, TILE_SIZE>(
        g, bucket, find_key, start_idx, key_pos, src_lane, bucket_max_size);

    if (occupy_result == OccupyResult::DUPLICATE) {
      copy_vector<V, TILE_SIZE>(g, bucket->vectors + key_pos * dim, find_value,
                                dim);
      if (rank == src_lane) {
        if (metas != nullptr) {
          *(metas + key_idx) =
              bucket->metas(key_pos)->load(cuda::std::memory_order_relaxed);
        }
        if (found != nullptr) {
          *(found + key_idx) = true;
        }
      }
    }
  }
}

template <typename K, typename V, typename M>
struct SelectLookupKernelWithIO {
  static void execute_kernel(const float& load_factor, const int& block_size,
                             const size_t bucket_max_size,
                             const size_t buckets_num, const size_t dim,
                             cudaStream_t& stream, const size_t& n,
                             const Table<K, V, M>* __restrict table,
                             const K* __restrict keys, V* __restrict values,
                             M* __restrict metas, bool* __restrict found) {
    if (load_factor <= 0.75) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      lookup_kernel_with_io<K, V, M, tile_size>
          <<<grid_size, block_size, 0, stream>>>(table, bucket_max_size,
                                                 buckets_num, dim, keys, values,
                                                 metas, found, N);
    } else {
      const unsigned int tile_size = 16;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      lookup_kernel_with_io<K, V, M, tile_size>
          <<<grid_size, block_size, 0, stream>>>(table, bucket_max_size,
                                                 buckets_num, dim, keys, values,
                                                 metas, found, N);
    }
    return;
  }
};

/* lookup kernel.
 */
template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__global__ void lookup_kernel(const Table<K, V, M>* __restrict table,
                              const size_t bucket_max_size,
                              const size_t buckets_num, const size_t dim,
                              const K* __restrict keys, V** __restrict values,
                              M* __restrict metas, bool* __restrict found,
                              int* __restrict dst_offset, size_t N) {
  int* buckets_size = table->buckets_size;
  Bucket<K, V, M>* buckets = table->buckets;

  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_idx = t / TILE_SIZE;

    const K find_key = keys[key_idx];
    if (IS_RESERVED_KEY(find_key)) continue;

    int key_pos = -1;
    int src_lane = -1;
    size_t bkt_idx = 0;
    size_t start_idx = 0;

    Bucket<K, V, M>* bucket = get_key_position<K>(
        buckets, find_key, bkt_idx, start_idx, buckets_num, bucket_max_size);

    const int bucket_size = buckets_size[bkt_idx];
    if (bucket_size >= bucket_max_size) {
      start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
    }

    if (dst_offset != nullptr && rank == 0) {
      *(dst_offset + key_idx) = key_idx;
    }

    OccupyResult occupy_result{OccupyResult::INITIAL};
    occupy_result = find_without_lock<K, V, M, TILE_SIZE>(
        g, bucket, find_key, start_idx, key_pos, src_lane, bucket_max_size);

    if (occupy_result == OccupyResult::DUPLICATE) {
      if (rank == src_lane) {
        *(values + key_idx) = (bucket->vectors + key_pos * dim);
        if (metas != nullptr) {
          *(metas + key_idx) =
              bucket->metas(key_pos)->load(cuda::std::memory_order_relaxed);
        }
        if (found != nullptr) {
          *(found + key_idx) = true;
        }
      }
    } else {
      if (rank == 0) {
        *(values + key_idx) = nullptr;
      }
    }
  }
}

/* lookup with IO operation. This kernel is
 * usually used for the pure HBM mode for better performance.
 */
template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__global__ void lookup_ptr_kernel(const Table<K, V, M>* __restrict table,
                                  const size_t bucket_max_size,
                                  const size_t buckets_num, const size_t dim,
                                  const K* __restrict keys,
                                  V** __restrict values, M* __restrict metas,
                                  bool* __restrict found, size_t N) {
  int* buckets_size = table->buckets_size;
  Bucket<K, V, M>* buckets = table->buckets;

  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_idx = t / TILE_SIZE;

    const K find_key = keys[key_idx];
    if (IS_RESERVED_KEY(find_key)) continue;

    int key_pos = -1;
    int src_lane = -1;
    size_t bkt_idx = 0;
    size_t start_idx = 0;

    Bucket<K, V, M>* bucket = get_key_position<K>(
        buckets, find_key, bkt_idx, start_idx, buckets_num, bucket_max_size);

    const int bucket_size = buckets_size[bkt_idx];
    if (bucket_size >= bucket_max_size) {
      start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
    }

    OccupyResult occupy_result{OccupyResult::INITIAL};
    occupy_result = find_without_lock<K, V, M, TILE_SIZE>(
        g, bucket, find_key, start_idx, key_pos, src_lane, bucket_max_size);

    if (occupy_result == OccupyResult::DUPLICATE) {
      if (rank == src_lane) {
        values[key_idx] = bucket->vectors + key_pos * dim;
        if (metas != nullptr) {
          *(metas + key_idx) =
              bucket->metas(key_pos)->load(cuda::std::memory_order_relaxed);
        }
        if (found != nullptr) {
          *(found + key_idx) = true;
        }
      }
    }
  }
}

template <typename K, typename V, typename M>
struct SelectLookupPtrKernel {
  static void execute_kernel(const float& load_factor, const int& block_size,
                             const size_t bucket_max_size,
                             const size_t buckets_num, const size_t dim,
                             cudaStream_t& stream, const size_t& n,
                             const Table<K, V, M>* __restrict table,
                             const K* __restrict keys, V** __restrict values,
                             M* __restrict metas, bool* __restrict found) {
    if (load_factor <= 0.75) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      lookup_ptr_kernel<K, V, M, tile_size>
          <<<grid_size, block_size, 0, stream>>>(table, bucket_max_size,
                                                 buckets_num, dim, keys, values,
                                                 metas, found, N);
    } else {
      const unsigned int tile_size = 16;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      lookup_ptr_kernel<K, V, M, tile_size>
          <<<grid_size, block_size, 0, stream>>>(table, bucket_max_size,
                                                 buckets_num, dim, keys, values,
                                                 metas, found, N);
    }
    return;
  }
};

/* Clear all key-value in the table. */
template <class K, class V, class M>
__global__ void clear_kernel(Table<K, V, M>* __restrict table, size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const size_t bucket_max_size = table->bucket_max_size;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int key_idx = t % bucket_max_size;
    int bkt_idx = t / bucket_max_size;
    Bucket<K, V, M>* bucket = &(table->buckets[bkt_idx]);

    (bucket->keys(key_idx))
        ->store(static_cast<K>(EMPTY_KEY), cuda::std::memory_order_relaxed);
    if (key_idx == 0) {
      table->buckets_size[bkt_idx] = 0;
    }
  }
}

/* Remove specified keys. */
template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__global__ void remove_kernel(const Table<K, V, M>* __restrict table,
                              const K* __restrict keys,
                              Bucket<K, V, M>* __restrict buckets,
                              int* __restrict buckets_size,
                              const size_t bucket_max_size,
                              const size_t buckets_num, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_idx = t / TILE_SIZE;
    K find_key = keys[key_idx];
    if (IS_RESERVED_KEY(find_key)) continue;

    int key_pos = -1;

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    uint32_t tile_offset = 0;

    Bucket<K, V, M>* bucket = get_key_position<K>(
        buckets, find_key, bkt_idx, start_idx, buckets_num, bucket_max_size);

    unsigned found_vote = 0;
#pragma unroll
    for (tile_offset = 0; tile_offset < bucket_max_size;
         tile_offset += TILE_SIZE) {
      key_pos = (start_idx + tile_offset + rank) & (bucket_max_size - 1);

      const K current_key =
          (bucket->keys(key_pos))->load(cuda::std::memory_order_relaxed);

      found_vote = g.ballot(find_key == current_key);
      if (found_vote) {
        break;
      }

      if (g.any(current_key == static_cast<K>(EMPTY_KEY))) {
        break;
      }
    }

    if (found_vote) {
      const int src_lane = __ffs(found_vote) - 1;

      if (g.thread_rank() == src_lane) {
        const int key_pos =
            (start_idx + tile_offset + src_lane) & (bucket_max_size - 1);
        (bucket->keys(key_pos))
            ->store(static_cast<K>(RECLAIM_KEY),
                    cuda::std::memory_order_relaxed);
        (bucket->metas(key_pos))
            ->store(static_cast<M>(EMPTY_META),
                    cuda::std::memory_order_relaxed);
        atomicSub(&buckets_size[bkt_idx], 1);
      }
      break;
    }
  }
}

/* Remove specified keys which match the Predict. */
template <class K, class V, class M, uint32_t TILE_SIZE = 1>
__global__ void remove_kernel(const Table<K, V, M>* __restrict table,
                              const EraseIfPredictInternal<K, M> pred,
                              const K pattern, const M threshold,
                              size_t* __restrict count,
                              Bucket<K, V, M>* __restrict buckets,
                              int* __restrict buckets_size,
                              const size_t bucket_max_size,
                              const size_t buckets_num, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    uint32_t bkt_idx = t;
    uint32_t key_pos = 0;

    Bucket<K, V, M>* bucket = buckets + bkt_idx;

    K current_key = 0;
    M current_meta = 0;
    uint32_t key_offset = 0;
    while (key_offset < bucket_max_size) {
      current_key =
          bucket->keys(key_offset)->load(cuda::std::memory_order_relaxed);
      current_meta =
          bucket->metas(key_offset)->load(cuda::std::memory_order_relaxed);
      if (!IS_RESERVED_KEY(current_key)) {
        if (pred(current_key, current_meta, pattern, threshold)) {
          atomicAdd(count, 1);
          key_pos = key_offset;
          (bucket->keys(key_pos))
              ->store(static_cast<K>(RECLAIM_KEY),
                      cuda::std::memory_order_relaxed);
          (bucket->metas(key_pos))
              ->store(static_cast<M>(EMPTY_META),
                      cuda::std::memory_order_relaxed);
          atomicSub(&buckets_size[bkt_idx], 1);
        } else {
          key_offset++;
        }
      } else {
        key_offset++;
      }
    }
  }
}

/* Dump with meta. */
template <class K, class V, class M>
inline std::tuple<size_t, size_t> dump_kernel_shared_memory_size(
    const size_t available_shared_memory) {
  const size_t block_size{std::min(
      available_shared_memory / 2 / sizeof(KVM<K, V, M>), UINT64_C(1024))};
  MERLIN_CHECK(
      block_size > 0,
      "[HierarchicalKV] block_size <= 0, the K-V-M size may be too large!");

  return {block_size * sizeof(KVM<K, V, M>), block_size};
}

template <class K, class V, class M>
__global__ void dump_kernel(const Table<K, V, M>* __restrict table, K* d_key,
                            V* __restrict d_val, M* __restrict d_meta,
                            const size_t offset, const size_t search_length,
                            size_t* d_dump_counter) {
  extern __shared__ unsigned char s[];
  KVM<K, V, M>* const block_tuples{reinterpret_cast<KVM<K, V, M>*>(s)};

  const size_t bucket_max_size{table->bucket_max_size};
  const size_t dim{table->dim};

  __shared__ size_t block_acc;
  __shared__ size_t global_acc;

  const size_t tid{blockIdx.x * blockDim.x + threadIdx.x};

  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  if (tid < search_length) {
    Bucket<K, V, M>* const bucket{
        &table->buckets[(tid + offset) / bucket_max_size]};

    const int key_idx{static_cast<int>((tid + offset) % bucket_max_size)};
    const K key{(bucket->keys(key_idx))->load(cuda::std::memory_order_relaxed)};

    if (!IS_RESERVED_KEY(key)) {
      size_t local_index{atomicAdd(&block_acc, 1)};
      block_tuples[local_index] = {
          key, &bucket->vectors[key_idx * dim],
          bucket->metas(key_idx)->load(cuda::std::memory_order_relaxed)};
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    global_acc = atomicAdd(d_dump_counter, block_acc);
  }
  __syncthreads();

  if (threadIdx.x < block_acc) {
    const KVM<K, V, M>& tuple{block_tuples[threadIdx.x]};

    const size_t j{global_acc + threadIdx.x};
    d_key[j] = tuple.key;
    for (int i{0}; i < dim; ++i) {
      d_val[j * dim + i] = tuple.value[i];
    }
    if (d_meta != nullptr) {
      d_meta[j] = tuple.meta;
    }
  }
}

/* Dump with meta. */
template <class K, class V, class M, template <typename, typename> class PredFunctor>
__global__ void dump_kernel(const Table<K, V, M>* __restrict table,
                            const K pattern, const M threshold, K* d_key,
                            V* __restrict d_val, M* __restrict d_meta,
                            const size_t offset, const size_t search_length,
                            size_t* d_dump_counter) {
  extern __shared__ unsigned char s[];
  const size_t bucket_max_size = table->bucket_max_size;
  const size_t dim = table->dim;
  K* smem = (K*)s;
  K* block_result_key = smem;
  V* block_result_val = (V*)&(smem[blockDim.x]);
  M* block_result_meta = (M*)&(block_result_val[blockDim.x * dim]);
  __shared__ size_t block_acc;
  __shared__ size_t global_acc;
  PredFunctor<K, M> fn;

  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();

  if (tid < search_length) {
    int bkt_idx = (tid + offset) / bucket_max_size;
    int key_idx = (tid + offset) % bucket_max_size;
    Bucket<K, V, M>* bucket = &(table->buckets[bkt_idx]);

    const K key =
        (bucket->keys(key_idx))->load(cuda::std::memory_order_relaxed);
    M meta = bucket->metas(key_idx)->load(cuda::std::memory_order_relaxed);

    if (key != static_cast<K>(EMPTY_KEY) &&
        fn(key, meta, pattern, threshold)) {
      size_t local_index = atomicAdd(&block_acc, 1);
      block_result_key[local_index] = key;
      for (int i = 0; i < dim; i++) {
        atomicExch(&(block_result_val[local_index * dim + i]),
                   bucket->vectors[key_idx * dim + i]);
      }
      if (d_meta != nullptr) {
        block_result_meta[local_index] = meta;
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    global_acc = atomicAdd(d_dump_counter, block_acc);
  }
  __syncthreads();

  if (threadIdx.x < block_acc) {
    d_key[global_acc + threadIdx.x] = block_result_key[threadIdx.x];
    for (int i = 0; i < dim; i++) {
      d_val[(global_acc + threadIdx.x) * dim + i] =
          block_result_val[threadIdx.x * dim + i];
    }
    if (d_meta != nullptr) {
      d_meta[global_acc + threadIdx.x] = block_result_meta[threadIdx.x];
    }
  }
}

/* If founds[i] = true, read data from corresponding address of
 * table_value_addrs and write to param_values; if founds[i] = false, write data
 * from param_values to corresponding address of table_value_addrs. usually
 * called by find_or_insert kernel.
 */
template <class V>
void read_or_write_by_cpu(V** __restrict table_value_addrs,
                          V* __restrict param_values,
                          const int* __restrict offset, const bool* founds,
                          size_t dim, int N, int n_worker = 16) {
  std::vector<std::thread> thds;
  if (n_worker < 1) n_worker = 1;

  auto functor = [founds, dim](V** __restrict table_value_addrs,
                               V* __restrict param_values,
                               const int* __restrict offset, int handled_size,
                               int trunk_size) -> void {
    for (int i = handled_size; i < handled_size + trunk_size; i++) {
      if (table_value_addrs[i] != nullptr) {
        if (founds[offset[i]]) {
          memcpy(param_values + offset[i] * dim, table_value_addrs[i],
                 sizeof(V) * dim);
        } else {
          memcpy(table_value_addrs[i], param_values + offset[i] * dim,
                 sizeof(V) * dim);
        }
      }
    }
  };

  int32_t trunk_size_floor = N / n_worker;
  int32_t trunk_size_remain = N % n_worker;
  int32_t n_worker_used = trunk_size_floor == 0 ? trunk_size_remain : n_worker;

  size_t handled_size = 0;
  for (int i = 0; i < n_worker_used; i++) {
    int32_t cur_trunk_size = trunk_size_floor;
    if (trunk_size_remain != 0) {
      cur_trunk_size += 1;
      trunk_size_remain--;
    }
    thds.push_back(std::thread(functor, table_value_addrs, param_values, offset,
                               handled_size, cur_trunk_size));
    handled_size += cur_trunk_size;
  }

  for (int i = 0; i < n_worker_used; i++) {
    thds[i].join();
  }
}

/*
 * find or insert with IO operation. This kernel is
 * usually used for the pure HBM mode for better performance.
 */
template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__global__ void find_or_insert_kernel_with_io(
    const Table<K, V, M>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    V* __restrict values, M* __restrict metas, const size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    const size_t key_idx = t / TILE_SIZE;

    const K find_or_insert_key = keys[key_idx];

    if (IS_RESERVED_KEY(find_or_insert_key)) continue;

    const M find_or_insert_meta =
        metas != nullptr ? metas[key_idx] : static_cast<M>(MAX_META);
    V* find_or_insert_value = values + key_idx * dim;

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;
    K evicted_key;

    Bucket<K, V, M>* bucket =
        get_key_position<K>(table->buckets, find_or_insert_key, bkt_idx,
                            start_idx, buckets_num, bucket_max_size);

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    do {
      if (bucket_size < bucket_max_size) {
        occupy_result = find_and_lock_when_vacant<K, V, M, TILE_SIZE>(
            g, bucket, find_or_insert_key, find_or_insert_meta, evicted_key,
            start_idx, key_pos, src_lane, bucket_max_size);
      } else {
        start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
        occupy_result = find_and_lock_when_full<K, V, M, TILE_SIZE>(
            g, bucket, find_or_insert_key, find_or_insert_meta, evicted_key,
            start_idx, key_pos, src_lane, bucket_max_size);
      }

      occupy_result = g.shfl(occupy_result, src_lane);
    } while (occupy_result == OccupyResult::CONTINUE);

    if (occupy_result == OccupyResult::REFUSED) continue;

    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        g.thread_rank() == src_lane) {
      atomicAdd(&(buckets_size[bkt_idx]), 1);
    }

    if (occupy_result == OccupyResult::DUPLICATE) {
      copy_vector<V, TILE_SIZE>(g, bucket->vectors + key_pos * dim,
                                find_or_insert_value, dim);
      if (metas != nullptr && g.thread_rank() == src_lane) {
        *(metas + key_idx) =
            bucket->metas(key_pos)->load(cuda::std::memory_order_relaxed);
      }
    } else {
      copy_vector<V, TILE_SIZE>(g, find_or_insert_value,
                                bucket->vectors + key_pos * dim, dim);
      if (g.thread_rank() == src_lane) {
        update_meta(bucket, key_pos, metas, key_idx);
      }
    }

    if (g.thread_rank() == src_lane) {
      (bucket->keys(key_pos))
          ->store(find_or_insert_key, cuda::std::memory_order_relaxed);
    }
  }
}

template <typename K, typename V, typename M>
struct SelectFindOrInsertKernelWithIO {
  static void execute_kernel(const float& load_factor, const int& block_size,
                             const size_t bucket_max_size,
                             const size_t buckets_num, const size_t dim,
                             cudaStream_t& stream, const size_t& n,
                             const Table<K, V, M>* __restrict table,
                             const K* __restrict keys, V* __restrict values,
                             M* __restrict metas) {
    if (load_factor <= 0.75) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      find_or_insert_kernel_with_io<K, V, M, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, values, metas, N);
    } else {
      const unsigned int tile_size = 32;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      find_or_insert_kernel_with_io<K, V, M, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, values, metas, N);
    }
    return;
  }
};

/* find or insert with the end-user specified meta.
 */
template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__global__ void find_or_insert_kernel(
    const Table<K, V, M>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    V** __restrict vectors, M* __restrict metas, bool* __restrict found,
    int* __restrict keys_index, const size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;

    const K find_or_insert_key = keys[key_idx];

    if (IS_RESERVED_KEY(find_or_insert_key)) continue;

    const M find_or_insert_meta =
        metas != nullptr ? metas[key_idx] : static_cast<M>(MAX_META);

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;
    K evicted_key;

    Bucket<K, V, M>* bucket =
        get_key_position<K>(table->buckets, find_or_insert_key, bkt_idx,
                            start_idx, buckets_num, bucket_max_size);

    if (g.thread_rank() == 0) {
      *(keys_index + key_idx) = key_idx;
    }

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    do {
      if (bucket_size < bucket_max_size) {
        occupy_result = find_and_lock_when_vacant<K, V, M, TILE_SIZE>(
            g, bucket, find_or_insert_key, find_or_insert_meta, evicted_key,
            start_idx, key_pos, src_lane, bucket_max_size);
      } else {
        start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
        occupy_result = find_and_lock_when_full<K, V, M, TILE_SIZE>(
            g, bucket, find_or_insert_key, find_or_insert_meta, evicted_key,
            start_idx, key_pos, src_lane, bucket_max_size);
      }

      occupy_result = g.shfl(occupy_result, src_lane);
    } while (occupy_result == OccupyResult::CONTINUE);

    if (occupy_result == OccupyResult::REFUSED) continue;

    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        g.thread_rank() == src_lane) {
      atomicAdd(&(buckets_size[bkt_idx]), 1);
    }

    if (occupy_result == OccupyResult::DUPLICATE) {
      if (g.thread_rank() == src_lane) {
        *(vectors + key_idx) = (bucket->vectors + key_pos * dim);

        if (found != nullptr) {
          *(found + key_idx) = true;
        }

        if (metas != nullptr) {
          *(metas + key_idx) =
              bucket->metas(key_pos)->load(cuda::std::memory_order_relaxed);
        }
      }
    } else {
      if (g.thread_rank() == src_lane) {
        *(vectors + key_idx) = (bucket->vectors + key_pos * dim);
        update_meta(bucket, key_pos, metas, key_idx);
      }
    }

    if (g.thread_rank() == src_lane) {
      (bucket->keys(key_pos))
          ->store(find_or_insert_key, cuda::std::memory_order_relaxed);
    }
  }
}

/* find or insert with the end-user specified meta.
 */
template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__global__ void find_ptr_or_insert_kernel(
    const Table<K, V, M>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    V** __restrict vectors, M* __restrict metas, bool* __restrict found,
    const size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;

    const K find_or_insert_key = keys[key_idx];

    if (IS_RESERVED_KEY(find_or_insert_key)) continue;

    const M find_or_insert_meta =
        metas != nullptr ? metas[key_idx] : static_cast<M>(MAX_META);

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;
    K evicted_key;

    Bucket<K, V, M>* bucket =
        get_key_position<K>(table->buckets, find_or_insert_key, bkt_idx,
                            start_idx, buckets_num, bucket_max_size);

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    do {
      if (bucket_size < bucket_max_size) {
        occupy_result = find_and_lock_when_vacant<K, V, M, TILE_SIZE>(
            g, bucket, find_or_insert_key, find_or_insert_meta, evicted_key,
            start_idx, key_pos, src_lane, bucket_max_size);
      } else {
        start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
        occupy_result = find_and_lock_when_full<K, V, M, TILE_SIZE>(
            g, bucket, find_or_insert_key, find_or_insert_meta, evicted_key,
            start_idx, key_pos, src_lane, bucket_max_size);
      }

      occupy_result = g.shfl(occupy_result, src_lane);
    } while (occupy_result == OccupyResult::CONTINUE);

    if (occupy_result == OccupyResult::REFUSED) continue;

    if ((occupy_result == OccupyResult::OCCUPIED_EMPTY ||
         occupy_result == OccupyResult::OCCUPIED_RECLAIMED) &&
        g.thread_rank() == src_lane) {
      atomicAdd(&(buckets_size[bkt_idx]), 1);
    }

    if (occupy_result == OccupyResult::DUPLICATE) {
      if (g.thread_rank() == src_lane) {
        *(vectors + key_idx) = (bucket->vectors + key_pos * dim);
        *(found + key_idx) = true;
        if (metas != nullptr) {
          *(metas + key_idx) =
              bucket->metas(key_pos)->load(cuda::std::memory_order_relaxed);
        }
      }
    } else {
      if (g.thread_rank() == src_lane) {
        *(vectors + key_idx) = (bucket->vectors + key_pos * dim);
        *(found + key_idx) = false;
        update_meta(bucket, key_pos, metas, key_idx);
      }
    }

    if (g.thread_rank() == src_lane) {
      (bucket->keys(key_pos))
          ->store(find_or_insert_key, cuda::std::memory_order_relaxed);
    }
  }
}

template <typename K, typename V, typename M>
struct SelectFindOrInsertPtrKernel {
  static void execute_kernel(const float& load_factor, const int& block_size,
                             const size_t bucket_max_size,
                             const size_t buckets_num, const size_t dim,
                             cudaStream_t& stream, const size_t& n,
                             const Table<K, V, M>* __restrict table,
                             const K* __restrict keys, V** __restrict values,
                             M* __restrict metas, bool* __restrict found) {
    if (load_factor <= 0.5) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      find_ptr_or_insert_kernel<K, V, M, tile_size>
          <<<grid_size, block_size, 0, stream>>>(table, bucket_max_size,
                                                 buckets_num, dim, keys, values,
                                                 metas, found, N);
    } else if (load_factor <= 0.875) {
      const unsigned int tile_size = 8;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      find_ptr_or_insert_kernel<K, V, M, tile_size>
          <<<grid_size, block_size, 0, stream>>>(table, bucket_max_size,
                                                 buckets_num, dim, keys, values,
                                                 metas, found, N);
    } else {
      const unsigned int tile_size = 32;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      find_ptr_or_insert_kernel<K, V, M, tile_size>
          <<<grid_size, block_size, 0, stream>>>(table, bucket_max_size,
                                                 buckets_num, dim, keys, values,
                                                 metas, found, N);
    }
    return;
  }
};

/* Read the data from address of table_value_addrs to corresponding position
  in param_value if mask[i] is true, otherwise write data to table_value_addrs
  form param_value,
  usually called by find_or_insert kernel.

  `table_value_addrs`: A pointer of pointer of V which should be on HBM,
        but each value (a pointer of V) could point to a
        memory on HBM or HMEM.
  `param_value`: A continue memory pointer with Vector
        which should be HBM.
  `mask`: One for each `param_value`. If true, reading from table_value_addrs,
          or false writing table_value_addrs from  param_value.
  `param_key_index`: N values from address of table_value_addrs are mapped to
        param_values according to param_key_index.
  `dim`: the dim of value.
  `N`: The number of vectors needed to be read.
*/
template <class K, class V, class M>
__global__ void read_or_write_kernel(V** __restrict table_value_addrs,
                                     V* __restrict param_values,
                                     const bool* mask,
                                     const int* __restrict param_key_index,
                                     const size_t dim, const size_t N) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    int vec_index = int(t / dim);
    int dim_index = t % dim;
    int real_key_index =
        param_key_index != nullptr ? param_key_index[vec_index] : vec_index;

    /// if found, read the value form table, otherwise write it
    if (table_value_addrs[vec_index] != nullptr) {
      /// find
      if (mask[real_key_index]) {
        param_values[real_key_index * dim + dim_index] =
            table_value_addrs[vec_index][dim_index];
      }
      /// insert
      else {
        table_value_addrs[vec_index][dim_index] =
            param_values[real_key_index * dim + dim_index];
      }
    }
  }
}

/*
 * update with IO operation. This kernel is
 * usually used for the pure HBM mode for better performance.
 */
template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__global__ void update_kernel_with_io(
    const Table<K, V, M>* __restrict table, const size_t bucket_max_size,
    const size_t buckets_num, const size_t dim, const K* __restrict keys,
    const V* __restrict values, const M* __restrict metas, const size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;

    const K update_key = keys[key_idx];

    if (IS_RESERVED_KEY(update_key)) continue;

    const V* update_value = values + key_idx * dim;

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;

    Bucket<K, V, M>* bucket =
        get_key_position<K>(table->buckets, update_key, bkt_idx, start_idx,
                            buckets_num, bucket_max_size);

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];

    if (bucket_size >= bucket_max_size) {
      start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
    }
    occupy_result = find_and_lock_for_update<K, V, M, TILE_SIZE>(
        g, bucket, update_key, start_idx, key_pos, src_lane, bucket_max_size);

    occupy_result = g.shfl(occupy_result, src_lane);

    if (occupy_result == OccupyResult::REFUSED) continue;

    if (occupy_result == OccupyResult::DUPLICATE) {
      copy_vector<V, TILE_SIZE>(g, update_value,
                                bucket->vectors + key_pos * dim, dim);
      if (src_lane == g.thread_rank()) {
        update_meta(bucket, key_pos, metas, key_idx);
      }
    }

    if (g.thread_rank() == src_lane) {
      (bucket->keys(key_pos))
          ->store(update_key, cuda::std::memory_order_relaxed);
    }
  }
}

template <typename K, typename V, typename M>
struct SelectUpdateKernelWithIO {
  static void execute_kernel(const float& load_factor, const int& block_size,
                             const size_t bucket_max_size,
                             const size_t buckets_num, const size_t dim,
                             cudaStream_t& stream, const size_t& n,
                             const Table<K, V, M>* __restrict table,
                             const K* __restrict keys,
                             const V* __restrict values,
                             const M* __restrict metas) {
    if (load_factor <= 0.75) {
      const unsigned int tile_size = 4;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      update_kernel_with_io<K, V, M, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, values, metas, N);
    } else {
      const unsigned int tile_size = 32;
      const size_t N = n * tile_size;
      const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
      update_kernel_with_io<K, V, M, tile_size>
          <<<grid_size, block_size, 0, stream>>>(
              table, bucket_max_size, buckets_num, dim, keys, values, metas, N);
    }
    return;
  }
};

template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__global__ void update_kernel(const Table<K, V, M>* __restrict table,
                              const size_t bucket_max_size,
                              const size_t buckets_num, const size_t dim,
                              const K* __restrict keys, V** __restrict vectors,
                              const M* __restrict metas,
                              int* __restrict src_offset, size_t N) {
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int* buckets_size = table->buckets_size;

  for (size_t t = (blockIdx.x * blockDim.x) + threadIdx.x; t < N;
       t += blockDim.x * gridDim.x) {
    int key_pos = -1;
    size_t key_idx = t / TILE_SIZE;

    const K update_key = keys[key_idx];

    if (IS_RESERVED_KEY(update_key)) continue;

    size_t bkt_idx = 0;
    size_t start_idx = 0;
    int src_lane = -1;

    Bucket<K, V, M>* bucket =
        get_key_position<K>(table->buckets, update_key, bkt_idx, start_idx,
                            buckets_num, bucket_max_size);

    OccupyResult occupy_result{OccupyResult::INITIAL};
    const int bucket_size = buckets_size[bkt_idx];
    *(src_offset + key_idx) = key_idx;

    if (bucket_size >= bucket_max_size) {
      start_idx = (start_idx / TILE_SIZE) * TILE_SIZE;
    }
    occupy_result = find_and_lock_for_update<K, V, M, TILE_SIZE>(
        g, bucket, update_key, start_idx, key_pos, src_lane, bucket_max_size);

    occupy_result = g.shfl(occupy_result, src_lane);

    if (occupy_result == OccupyResult::REFUSED) continue;

    if (g.thread_rank() == src_lane) {
      if (occupy_result == OccupyResult::DUPLICATE) {
        *(vectors + key_idx) = (bucket->vectors + key_pos * dim);
        update_meta(bucket, key_pos, metas, key_idx);
      } else {
        *(vectors + key_idx) = nullptr;
      }
    }

    if (g.thread_rank() == src_lane) {
      (bucket->keys(key_pos))
          ->store(update_key, cuda::std::memory_order_relaxed);
    }
  }
}

}  // namespace merlin
}  // namespace nv
