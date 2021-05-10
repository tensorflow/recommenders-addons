/* Copyright 2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#ifndef NV_ALLOCATOR_H_
#define NV_ALLOCATOR_H_
#include "cub/util_allocator.cuh"
#include "cuda_runtime_api.h"
#include "nv_util.h"

namespace nv {

class CubAllocator {
 public:
  CubAllocator() : allocator_(8u, 3u) {}
  CubAllocator(const int*, int) : allocator_(8u, 3u) {}

  void malloc(void** ptr, size_t size, cudaStream_t stream = 0) {
    CUDA_CHECK(allocator_.DeviceAllocate(ptr, size, stream));
  }
  void malloc(void** ptr, size_t size, int dev, cudaStream_t stream = 0) {
    CudaDeviceRestorer dev_restorer;
    CUDA_CHECK(allocator_.DeviceAllocate(dev, ptr, size, stream));
  }

  void free(void* ptr) { CUDA_CHECK(allocator_.DeviceFree(ptr)); }
  void free(void* ptr, int dev) {
    CudaDeviceRestorer dev_restorer;
    CUDA_CHECK(allocator_.DeviceFree(dev, ptr));
  }

 private:
  cub::CachingDeviceAllocator allocator_;
};

}  // namespace nv

#endif
