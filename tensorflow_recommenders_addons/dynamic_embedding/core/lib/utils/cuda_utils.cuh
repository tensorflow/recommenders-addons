#ifndef CUDA_UTILS_CUH_
#define CUDA_UTILS_CUH_

#include <stdint.h>
#include <stdio.h>
#include <exception>
#include <stdexcept>
#include <string>
#include "cuda_runtime.h"
#include "nvml.h"

#include "tensorflow/core/framework/types.h"

// TODO: can we do this more efficiently?
__inline__ __device__ int8_t atomicCAS(int8_t* address, int8_t compare, int8_t val)
{
  int32_t *base_address = (int32_t*)((char*)address - ((size_t)address & 3));
  int32_t int_val = (int32_t)val << (((size_t)address & 3) * 8);
  int32_t int_comp = (int32_t)compare << (((size_t)address & 3) * 8);
  return (int8_t)atomicCAS(base_address, int_comp, int_val);
}

// TODO: can we do this more efficiently?
/*__inline__ __device__ int16_t atomicCAS(int16_t* address, int16_t compare, int16_t val)
{
  int32_t *base_address = (int32_t*)((char*)address - ((size_t)address & 2));
  int32_t int_val = (int32_t)val << (((size_t)address & 2) * 8);
  int32_t int_comp = (int32_t)compare << (((size_t)address & 2) * 8);
  return (int16_t)atomicCAS(base_address, int_comp, int_val);
}*/

__inline__ __device__ int64_t atomicCAS(int64_t* address, int64_t compare, int64_t val)
{
  return (int64_t)atomicCAS((unsigned long long*)address, (unsigned long long)compare, (unsigned long long)val);
}

__inline__ __device__ uint64_t atomicCAS(uint64_t* address, uint64_t compare, uint64_t val)
{
  return (uint64_t)atomicCAS((unsigned long long*)address, (unsigned long long)compare, (unsigned long long)val);
}

__inline__ __device__ long long int atomicCAS(long long int* address, long long int compare, long long int val)
{
  return (long long int)atomicCAS((unsigned long long*)address, (unsigned long long)compare, (unsigned long long)val);
}

__inline__ __device__ double atomicCAS(double* address, double compare, double val)
{
  return __longlong_as_double(atomicCAS((unsigned long long int*)address, __double_as_longlong(compare), __double_as_longlong(val)));
}

__inline__ __device__ float atomicCAS(float* address, float compare, float val)
{
  return __int_as_float(atomicCAS((int*)address, __float_as_int(compare), __float_as_int(val)));
}

__inline__ __device__ int64_t atomicAdd(int64_t* address, const int64_t val)
{
  return (int64_t) atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

__inline__ __device__ uint64_t atomicAdd(uint64_t* address,  const uint64_t val)
{
  return (uint64_t) atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

__inline__ __device__ signed char atomicAdd(signed char* address, const signed char val)
{
  int *base_address = (int*)((char*)address - ((size_t)address & 3));
  int int_val = (int)val << (((size_t)address & 3) * 8);

  return (signed char) atomicAdd((int*)base_address, (int)int_val);
}

class CudaException : public std::runtime_error {
 public:
  CudaException(const std::string& what) : runtime_error(what) {}
};

namespace nv_util {

inline void cuda_check_(cudaError_t val, const char* file, int line) {
  if (val != cudaSuccess) {
    throw CudaException(std::string(file) + ":" + std::to_string(line) +
                        ": CUDA error " + std::to_string(val) + ": " +
                        cudaGetErrorString(val));
  }
}

#define CUDA_CHECK(val) \
  { nv_util::cuda_check_((val), __FILE__, __LINE__); }

inline void __cudaCheckErrorHost(const char* file, const int line) {
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
            cudaGetErrorString(err));
    exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  err = cudaDeviceSynchronize();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file,
            line, cudaGetErrorString(err));
    exit(-1);
  }
  return;
}

//#define CudaCheckErrorHost() nv_util::__cudaCheckErrorHost(__FILE__, __LINE__)
#define CudaCheckErrorHost() {}

#ifndef NVML_CHECK
#define NVML_CHECK( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<nvmlReturn_t>( call );                                                               \
        if ( status != NVML_SUCCESS )                                                                                  \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA NVML call \"%s\" in line %d of file %s failed "                                      \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     nvmlErrorString( status ),                                                                        \
                     status );                                                                                         \
    }
#endif  // NVML_CHECK

}  // namespace nv_util

#endif  // CUDA_UTILS_CUH_
