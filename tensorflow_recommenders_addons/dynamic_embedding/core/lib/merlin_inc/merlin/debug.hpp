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

#include <cuda_runtime_api.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include "tensorflow_recommenders_addons/dynamic_embedding/core/lib/utils/cuda_utils.cuh"

namespace nv {
namespace merlin {

class CudaException : public std::runtime_error {
 public:
  CudaException(const std::string& what) : runtime_error(what) {}
};

inline void cuda_check_(cudaError_t val, const char* file, int line) {
  if (val != cudaSuccess) {
    std::ostringstream os;
    os << file << ':' << line << ": CUDA error " << cudaGetErrorName(val)
       << " (#" << val << "): " << cudaGetErrorString(val);
    throw CudaException(os.str());
  }
}

//#ifdef CUDA_CHECK
//#error Unexpected redfinition of CUDA_CHECK! Something is wrong.
//#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(val)                                 \
  do {                                                  \
    nv::merlin::cuda_check_((val), __FILE__, __LINE__); \
  } while (0)
#endif  // CUDA_CHECK

class MerlinException : public std::runtime_error {
 public:
  MerlinException(const std::string& what) : runtime_error(what) {}
};

template <class Msg>
inline void merlin_check_(bool cond, const Msg& msg, const char* file,
                          int line) {
  if (!cond) {
    std::ostringstream os;
    os << file << ':' << line << ": HierarchicalKV error " << msg;
    throw MerlinException(os.str());
  }
}

#ifdef MERLIN_CHECK
#error Unexpected redfinition of MERLIN_CHECK! Something is wrong.
#endif

#define MERLIN_CHECK(cond, msg)                                   \
  do {                                                            \
    nv::merlin::merlin_check_((cond), (msg), __FILE__, __LINE__); \
  } while (0)

}  // namespace merlin
}  // namespace nv
