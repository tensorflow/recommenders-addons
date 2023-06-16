/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "tensorflow_recommenders_addons/dynamic_embedding/core/lib/utils/cuda_utils.cuh"

#include <cooperative_groups.h>
#include "cuda_runtime.h"
#include "thrust/device_vector.h"
#include "thrust/execution_policy.h"
#include "thrust/scan.h"
#include "thrust/count.h"
#include "types.cuh"
#include "utils.cuh"

namespace nv {
namespace merlin {

template <typename K>
__global__ void keys_not_empty(const K* keys, bool* masks, size_t n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    masks[tid] = keys[tid] != EMPTY_KEY;
  }
}

template <typename Tidx, int TILE_SIZE = 8>
__global__ void gpu_cell_count(const bool* masks, bool target,
                               Tidx* offsets, size_t n, size_t* n_existed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();
  bool is_existed = false;
  if (tid < n) {
    if (masks[tid] == target) {
      is_existed = true;
    }
  }
  unsigned int vote = g.ballot(is_existed);
  int g_ones = __popc((int)vote);
  if (rank == 0 && tid < n) {
    offsets[tid / TILE_SIZE] = static_cast<Tidx>(g_ones);
    atomicAdd(static_cast<size_t*>(n_existed), static_cast<size_t>(g_ones));
  }
}

template <typename K, typename Tidx, int TILE_SIZE = 8>
__global__ void gpu_select_key_kernel(const bool* masks, bool target, size_t n,
                                      const Tidx* offsets, const K* __restrict keys,
                                      K* __restrict outkeys, Tidx* outoffsets) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  bool is_existed = false;
  if (tid < n) {
    if (masks[tid] == target) {
      is_existed = true;
    }
  }
  unsigned int vote = g.ballot(is_existed);
  unsigned int r_vote = __brev(vote) >> (32 - TILE_SIZE);

  if (tid < n) {
    r_vote = r_vote >> (TILE_SIZE - rank - 1);
    if (masks[tid] == target) {
      int prefix_n = __popc(r_vote) - 1;
      Tidx bias = offsets[tid / TILE_SIZE] + static_cast<Tidx>(prefix_n);
      outkeys[bias] = keys[tid];
      outoffsets[bias] = static_cast<Tidx>(tid);
    }
  }
}

template <typename K, typename V, typename Tidx, int TILE_SIZE = 8>
__global__ void gpu_select_kv_kernel(const bool* masks, bool target, size_t n,
                                      const Tidx* offsets,
                                      const K* __restrict keys,
                                      V* __restrict values,
                                      K* __restrict outkeys,
                                      V* __restrict outvalues,
                                      const size_t dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  bool is_existed = false;
  if (tid < n) {
    if (masks[tid] == target) {
      is_existed = true;
    }
  }
  unsigned int vote = g.ballot(is_existed);
  unsigned int r_vote = __brev(vote) >> (32 - TILE_SIZE);

  if (tid < n) {
    r_vote = r_vote >> (TILE_SIZE - rank - 1);
    if (masks[tid] == target) {
      int prefix_n = __popc(r_vote) - 1;
      Tidx bias = offsets[tid / TILE_SIZE] + static_cast<Tidx>(prefix_n);
      outkeys[bias] = keys[tid];
      for (size_t i=0;i<dim;i++) {
        outvalues[dim * bias + i] = values[dim * tid + i];
      }
    }
  }
}

template <typename K, typename V, typename M, typename Tidx, int TILE_SIZE = 8>
__global__ void gpu_select_kvm_kernel(const bool* masks, size_t n,
                                      const Tidx* offsets, K* __restrict keys,
                                      V* __restrict values, M* __restrict metas,
                                      const size_t dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  bool is_existed = false;
  if (tid < n) {
    if (masks[tid]) {
      is_existed = true;
    }
  }
  unsigned int vote = g.ballot(is_existed);
  unsigned int r_vote = __brev(vote) >> (32 - TILE_SIZE);
  K empty_key = (K)EMPTY_KEY;

  if (tid < n) {
    r_vote = r_vote >> (TILE_SIZE - rank - 1);
    if (masks[tid]) {
      int prefix_n = __popc(r_vote) - 1;
      Tidx bias = offsets[tid / TILE_SIZE] + static_cast<Tidx>(prefix_n);

      if (bias == tid) return;

      K target_key = 0;
      AtomicKey<K>* atomic_key = reinterpret_cast<AtomicKey<K>*>(keys) + bias;
      while (target_key != empty_key) {
        //target_key = atomicCAS(keys + bias, empty_key, keys[tid]);
        target_key = empty_key;
        atomic_key->compare_exchange_weak(target_key, keys[tid],
                                          cuda::std::memory_order_relaxed,
                                          cuda::std::memory_order_relaxed);
      }
      if (metas) metas[bias] = metas[tid];
      for (size_t j = 0; j < dim; j++) {
        values[dim * bias + j] = values[dim * tid + j];
      }
      //atomicExch(keys + tid, empty_key);
      atomic_key = reinterpret_cast<AtomicKey<K>*>(keys) + tid;
      atomic_key->store(empty_key, cuda::std::memory_order_relaxed);
    }
  }
}

template <typename K, typename V, typename M, int TILE_SIZE = 8>
__global__ void gpu_select_kvm_kernel_v2(size_t n,
                                         K* __restrict keys,
                                         V* __restrict values,
                                         M* __restrict metas,
                                         K* __restrict tmp_keys,
                                         V* __restrict tmp_values,
                                         M* __restrict tmp_metas,
                                         size_t* cnt,
                                         const size_t dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    size_t offset = atomicAdd(cnt, 1llu);
    tmp_keys[offset] = keys[tid];
    for (size_t j = 0; j < dim; j++) {
      tmp_values[offset * dim + j] = values[tid * dim + j];
    }
    if (metas) {
      tmp_metas[offset] = metas[tid];
    }
  }
}

template <typename K, typename V, typename M, typename Tidx, int TILE_SIZE = 8>
void gpu_pick_kvm_inplace(size_t grid_size, size_t block_size, const bool* masks,
                      bool target, size_t n, size_t* n_evicted, Tidx* offsets,
                      K* __restrict keys, V* __restrict values,
                      M* __restrict metas, size_t dim, cudaStream_t stream) {
  size_t n_offsets = (n + TILE_SIZE - 1) / TILE_SIZE;
  gpu_cell_count<Tidx, TILE_SIZE>
      <<<grid_size, block_size, 0, stream>>>(masks, target, offsets, n, n_evicted);
#if THRUST_VERSION >= 101600
  auto policy = thrust::cuda::par_nosync.on(stream);
#else
  auto policy = thrust::cuda::par.on(stream);
#endif
  thrust::device_ptr<Tidx> d_src(offsets);
  thrust::device_ptr<Tidx> d_dest(offsets);
  thrust::exclusive_scan(policy, d_src, d_src + n_offsets, d_dest);
  if (target) {
    gpu_select_kvm_kernel<K, V, M, Tidx, TILE_SIZE>
        <<<grid_size, block_size, 0, stream>>>(masks, n, offsets,
                keys, values, metas, dim);
  } else {
    throw std::runtime_error("Not used");
    //gpu_select_kvm_kernel_reverse<K, V, M, Tidx, TILE_SIZE>
    //    <<<grid_size, block_size, 0, stream>>>(masks, n, offsets,
    //            keys, values, metas, dim);
  }
}

template <typename K, typename V, typename M, int TILE_SIZE = 8>
size_t gpu_pick_kvm_v2(size_t grid_size, size_t block_size,
                      bool target, size_t n, size_t* n_evicted,
                      K* __restrict keys, V* __restrict values,
                      M* __restrict metas, size_t dim, cudaStream_t stream) {
#if THRUST_VERSION >= 101600
  auto policy = thrust::cuda::par_nosync.on(stream);
#else
  auto policy = thrust::cuda::par.on(stream);
#endif
  thrust::device_ptr<K> d_src(keys);
  int empty_cnt = thrust::count(policy, d_src, d_src + n, (K)EMPTY_KEY);
  size_t h_cnt = n - static_cast<size_t>(empty_cnt);
  if (h_cnt == 0) {
    return 0;
  }
  K* tmp_keys = nullptr;
  V* tmp_values = nullptr;
  M* tmp_metas = nullptr;
  if (target) {
    CUDA_CHECK(cudaMallocAsync(&tmp_keys, h_cnt * sizeof(K), stream));
    CUDA_CHECK(cudaMemsetAsync(tmp_keys, 0, h_cnt * sizeof(K), stream));
    CUDA_CHECK(cudaMallocAsync(&tmp_values, h_cnt * dim * sizeof(V), stream));
    CUDA_CHECK(cudaMemsetAsync(tmp_values, 0, h_cnt * dim * sizeof(V), stream));
    if (metas) {
      CUDA_CHECK(cudaMallocAsync(&tmp_metas, h_cnt * sizeof(M), stream));
      CUDA_CHECK(cudaMemsetAsync(tmp_metas, 0, h_cnt * sizeof(M), stream));
    }
    gpu_select_kvm_kernel_v2<K, V, M, TILE_SIZE>
        <<<grid_size, block_size, 0, stream>>>(n,
                keys, values, metas, tmp_keys, tmp_values, tmp_metas, n_evicted, dim);
    CUDA_CHECK(cudaMemcpyAsync(keys, tmp_keys, h_cnt * sizeof(K), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(values, tmp_values, h_cnt * dim * sizeof(V), cudaMemcpyDeviceToDevice, stream));
    if(metas) {
      CUDA_CHECK(cudaMemcpyAsync(metas, tmp_metas, h_cnt * sizeof(M), cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaFreeAsync(tmp_keys, stream));
    CUDA_CHECK(cudaFreeAsync(tmp_values, stream));
    if (tmp_metas) {
      CUDA_CHECK(cudaFreeAsync(tmp_metas, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } else {
    throw std::runtime_error("Not used");
    //gpu_select_kvm_kernel_reverse<K, V, M, Tidx, TILE_SIZE>
    //    <<<grid_size, block_size, 0, stream>>>(masks, n, offsets,
    //            keys, values, metas, dim);
  }
  return h_cnt;
}

template <typename K, typename V, typename M, int TILE_SIZE = 8>
void gpu_pick_kvm_inplace_wrap(const bool* masks, bool target,
                           size_t n, size_t* n_evicted,
                           K* __restrict keys, V* __restrict values,
                           M* __restrict metas, size_t dim, cudaStream_t stream) {
  size_t block_size = 256;
  size_t grid_size = SAFE_GET_GRID_SIZE(n, block_size);
  size_t n_offsets = (n + TILE_SIZE - 1) / TILE_SIZE;
  int64_t* offsets = nullptr;
  CUDA_CHECK(cudaMallocAsync(&offsets, sizeof(int64_t) * n_offsets, stream));
  gpu_pick_kvm_inplace<K, V, M, int64_t, TILE_SIZE>(grid_size, block_size,
          masks, target, n, n_evicted, offsets, keys, values, metas, dim, stream);
  CUDA_CHECK(cudaFreeAsync(offsets, stream));
}

template <typename K, int TILE_SIZE = 32>
void gpu_pick_keys(const bool* masks, bool target, size_t n, size_t* n_evicted,
                           const K* __restrict keys, K* __restrict outkeys,
                           int64_t* outoffsets, cudaStream_t stream) {
  size_t block_size = 256;
  size_t grid_size = SAFE_GET_GRID_SIZE(n, block_size);
  size_t n_offsets = (n + TILE_SIZE - 1) / TILE_SIZE;
  int64_t* offsets = nullptr;
  CUDA_CHECK(cudaMallocAsync(&offsets, sizeof(int64_t) * n_offsets, stream));
  CUDA_CHECK(cudaMemsetAsync(offsets, 0, sizeof(int64_t) * n_offsets, stream));

  gpu_cell_count<int64_t, TILE_SIZE>
      <<<grid_size, block_size, 0, stream>>>(masks, target, offsets, n, n_evicted);
#if THRUST_VERSION >= 101600
  auto policy = thrust::cuda::par_nosync.on(stream);
#else
  auto policy = thrust::cuda::par.on(stream);
#endif
  thrust::device_ptr<int64_t> d_src(offsets);
  thrust::device_ptr<int64_t> d_dest(offsets);
  thrust::exclusive_scan(policy, d_src, d_src + n_offsets, d_dest);
  gpu_select_key_kernel<K, int64_t, TILE_SIZE>
      <<<grid_size, block_size, 0, stream>>>(masks, target, n, offsets,
              keys, outkeys, outoffsets);
  CUDA_CHECK(cudaFreeAsync(offsets, stream));
}

template <typename K, typename V, int TILE_SIZE = 8>
void gpu_pick_kvs(const bool* masks, bool target, size_t n, size_t* n_evicted,
                           const K* __restrict keys,
                           V* __restrict values,
                           K* __restrict outkeys,
                           V* __restrict outvalues,
                           size_t dim,
                           cudaStream_t stream) {
  size_t block_size = 256;
  size_t grid_size = SAFE_GET_GRID_SIZE(n, block_size);
  size_t n_offsets = (n + TILE_SIZE - 1) / TILE_SIZE;
  int64_t* offsets = nullptr;
  CUDA_CHECK(cudaMallocAsync(&offsets, sizeof(int64_t) * n_offsets, stream));
  CUDA_CHECK(cudaMemsetAsync(offsets, 0, sizeof(int64_t) * n_offsets, stream));

  gpu_cell_count<int64_t, TILE_SIZE>
      <<<grid_size, block_size, 0, stream>>>(masks, target, offsets, n, n_evicted);
#if THRUST_VERSION >= 101600
  auto policy = thrust::cuda::par_nosync.on(stream);
#else
  auto policy = thrust::cuda::par.on(stream);
#endif
  thrust::device_ptr<int64_t> d_src(offsets);
  thrust::device_ptr<int64_t> d_dest(offsets);
  thrust::exclusive_scan(policy, d_src, d_src + n_offsets, d_dest);
  gpu_select_kv_kernel<K, V, int64_t, TILE_SIZE>
      <<<grid_size, block_size, 0, stream>>>(masks, target, n, offsets,
              keys, values, outkeys, outvalues, dim);
  CUDA_CHECK(cudaFreeAsync(offsets, stream));
}
}  // namespace merlin
}  // namespace nv
