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

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "curand_philox4x32_x.h"
#include "types.cuh"
#include "utils.cuh"

namespace nv {
namespace merlin {
namespace initializers {

inline void cuda_rand_check_(curandStatus_t val, const char* file, int line) {
  if (val != CURAND_STATUS_SUCCESS) {
    throw CudaException(std::string(file) + ":" + std::to_string(line) +
                        ": CURAND error " + std::to_string(val));
  }
}

#define CURAND_CHECK(val) \
  { nv::merlin::initializers::cuda_rand_check_((val), __FILE__, __LINE__); }

template <class T>
void zeros(T* d_data, const size_t len, cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(d_data, 0, len, stream));
}

template <class T>
void random_normal(T* d_data, const size_t len, cudaStream_t stream,
                   const T mean = 0.0, const T stddev = 0.05,
                   const unsigned long long seed = 2022ULL) {
  curandGenerator_t generator;
  CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, seed));
  CURAND_CHECK(curandGenerateNormal(generator, d_data, len, mean, stddev));
}

template <class T>
__global__ void adjust_max_min(T* d_data, const T minval, const T maxval,
                               const size_t N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) {
    d_data[tid] =
        d_data[tid] * (maxval - minval) + (0.5 * (maxval + minval) - 0.5);
  }
}

template <class T>
void random_uniform(T* d_data, const size_t len, cudaStream_t stream,
                    const T minval = 0.0, const T maxval = 1.0,
                    const unsigned long long seed = 2022ULL) {
  curandGenerator_t generator;

  CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, seed));

  int N = len;
  int block_size = 256;
  int grid_size = (N + block_size - 1) / block_size;
  CURAND_CHECK(curandGenerateUniform(generator, d_data, N));
  adjust_max_min<T>
      <<<grid_size, block_size, 0, stream>>>(d_data, minval, maxval, N);
}

template <class T>
__global__ void init_states(curandStatePhilox4_32_10_t* states,
                            const unsigned long long seed, const size_t N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) {
    curand_init(seed, tid, 0, &states[tid]);
  }
}

template <class T>
__global__ void make_truncated_normal(T* d_data,
                                      curandStatePhilox4_32_10_t* states,
                                      const size_t N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) {
    constexpr T truncated_val = T(2.0);
    while (fabsf(d_data[tid]) > truncated_val) {
      d_data[tid] = curand_normal(&states[tid]);
    }
  }
}

template <class T>
void truncated_normal(T* d_data, const size_t len, cudaStream_t stream,
                      const T minval = 0.0, const T maxval = 1.0,
                      const unsigned long long seed = 2022ULL) {
  curandGenerator_t generator;

  CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, seed));

  int N = len;
  int block_size = 256;
  int grid_size = (N + block_size - 1) / block_size;
  curandStatePhilox4_32_10_t* d_states;
  CUDA_CHECK(cudaMallocAsync(&d_states, N, stream));

  init_states<T><<<grid_size, block_size, 0, stream>>>(d_states, seed, N);

  make_truncated_normal<T>
      <<<grid_size, block_size, 0, stream>>>(d_data, d_states, N);

  adjust_max_min<T>
      <<<grid_size, block_size, 0, stream>>>(d_data, minval, maxval, N);

  CUDA_CHECK(cudaFreeAsync(d_states, stream));
}

template <class T>
class Initializer {
 public:
  virtual ~Initializer() {}
  virtual void initialize(T* data, size_t len, cudaStream_t stream) {}
};

template <class T>
class Zeros final : public Initializer<T> {
 public:
  void initialize(T* data, const size_t len, cudaStream_t stream) override {
    zeros<T>(data, len, stream);
  }
};

}  // namespace initializers
}  // namespace merlin
}  // namespace nv