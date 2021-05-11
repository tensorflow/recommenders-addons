/* Copyright 2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#ifndef NV_UTIL_H_
#define NV_UTIL_H_
#include <exception>
#include <string>

#include "cuda_runtime_api.h"

#define CUDA_CHECK(val) \
  { nv::cuda_check_((val), __FILE__, __LINE__); }

namespace nv {

class CudaException : public std::runtime_error {
 public:
  CudaException(const std::string& what) : runtime_error(what) {}
};

inline void cuda_check_(cudaError_t val, const char* file, int line) {
  if (val != cudaSuccess) {
    throw CudaException(std::string(file) + ":" + std::to_string(line) +
                        ": CUDA error " + std::to_string(val) + ": " +
                        cudaGetErrorString(val));
  }
}

class CudaDeviceRestorer {
 public:
  CudaDeviceRestorer() { CUDA_CHECK(cudaGetDevice(&dev_)); }
  ~CudaDeviceRestorer() { CUDA_CHECK(cudaSetDevice(dev_)); }

 private:
  int dev_;
};

inline int get_dev(const void* ptr) {
  cudaPointerAttributes attr;
  CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
  int dev = -1;

#if CUDART_VERSION >= 10000
  if (attr.type == cudaMemoryTypeDevice)
#else
  if (attr.memoryType == cudaMemoryTypeDevice)
#endif
  {
    dev = attr.device;
  }
  return dev;
}

inline void switch_to_dev(const void* ptr) {
  int dev = get_dev(ptr);
  if (dev >= 0) {
    CUDA_CHECK(cudaSetDevice(dev));
  }
}

}  // namespace nv
#endif
