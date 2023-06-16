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
#include "types.cuh"
#include "utils.cuh"

namespace nv {
namespace merlin {
namespace optimizers {

template <class T>
__global__ void adam_update_kernel(int len, float* weight, T* m, T* v,
                                   const T* wgrad, float alpha_t, float beta1,
                                   float beta2, float epsilon, float scaler) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float gi = TypeConvertFunc<float, T>::convert(wgrad[i]) / scaler;
    float mi =
        beta1 * TypeConvertFunc<float, T>::convert(m[i]) + (1.f - beta1) * gi;
    float vi = beta2 * TypeConvertFunc<float, T>::convert(v[i]) +
               (1.f - beta2) * gi * gi;
    m[i] = TypeConvertFunc<T, float>::convert(mi);
    v[i] = TypeConvertFunc<T, float>::convert(vi);
    weight[i] -= alpha_t * mi / (sqrt(vi) + epsilon);
  }
}

template <class T>
__global__ void ada_grad_update_kernel(int len, float* weight, const T* wgrad,
                                       T* sum, float lr, const float epsilon,
                                       float scaler) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float gi = TypeConvertFunc<float, T>::convert(wgrad[i]) / scaler;
    float accum_ = TypeConvertFunc<float, T>::convert(__ldg(&sum[i]));
    accum_ += gi * gi;
    float std_ = epsilon + sqrtf(accum_);
    weight[i] -= lr * gi / std_;
    sum[i] = TypeConvertFunc<T, float>::convert(accum_);
  }
}

template <class T>
__global__ void momentum_sgd_update_kernel(int len, float* weight, T* momentum,
                                           const T* wgrad, float lr,
                                           float momentum_factor,
                                           float scaler) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < len) {
    float mv =
        momentum_factor * TypeConvertFunc<float, T>::convert(momentum[idx]) -
        lr * TypeConvertFunc<float, T>::convert(wgrad[idx]) / scaler;
    momentum[idx] = TypeConvertFunc<T, float>::convert(mv);
    weight[idx] += mv;
  }
  return;
}

}  // namespace optimizers
}  // namespace merlin
}  // namespace nv