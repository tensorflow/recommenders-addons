/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include <cuda_runtime.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace {

template <typename T>
__global__ void TfraDynamicStitchKernel(
    const int32 slice_size, const int32 output_size,
    GpuDeviceArrayStruct<int32> input_indices,
    GpuDeviceArrayStruct<const T*> input_ptrs, T* output) {
  int32* data_indices = GetGpuDeviceArrayOnDevice(&input_indices);
  const T** data_ptrs = GetGpuDeviceArrayOnDevice(&input_ptrs);
  GPU_1D_KERNEL_LOOP(output_index, output_size) {
    const int32 slice_id = output_index / slice_size;
    const int32 slice_offset = output_index % slice_size;
    const int32 input_index = data_indices[slice_id];
    if (input_index != -1) {
      output[output_index] = ldg(data_ptrs[input_index] + slice_offset);
    }
  }
}

template <typename T>
__global__ void TfraDynamicStitchFastKernel(
    const int32 slice_size, const int32 indices_nsegments,
    GpuDeviceArrayStruct<int32> indices_vec_sizes,
    GpuDeviceArrayStruct<const int32*> indices_inputs_base,
    GpuDeviceArrayStruct<const T*> data_inputs_base, T* output) {
  const int32* indices_vec_sizes_ptr =
      GetGpuDeviceArrayOnDevice(&indices_vec_sizes);
  const int32** indices_inputs_base_ptr =
      GetGpuDeviceArrayOnDevice(&indices_inputs_base);
  const T** data_inputs_base_ptr = GetGpuDeviceArrayOnDevice(&data_inputs_base);

  __shared__ int32 indices_vec_size;
  __shared__ int32* indices_vec;
  __shared__ T* data_ptr_base;

  for (int32 indice_seg : GpuGridRangeY(indices_nsegments)) {
    indices_vec_size = indices_vec_sizes_ptr[indice_seg];
    indices_vec = const_cast<int32*>(indices_inputs_base_ptr[indice_seg]);
    data_ptr_base = const_cast<T*>(data_inputs_base_ptr[indice_seg]);
    for (int32 indice_i : GpuGridRangeX(indices_vec_size)) {
      const int32 output_indice_base = indices_vec[indice_i] * slice_size;
      const int32 data_ptr_indice_base = indice_i * slice_size;
#pragma unroll
      for (int32 slice_offset = 0; slice_offset < slice_size; slice_offset++) {
        output[output_indice_base + slice_offset] =
            data_ptr_base[data_ptr_indice_base + slice_offset];
      }
    }
  }
}

}  // namespace

template <typename T>
void TfraDynamicStitchGPUImpl(const Eigen::GpuDevice& gpu_device,
                              const int32 slice_size,
                              const int32 first_dim_size,
                              const GpuDeviceArrayStruct<int>& input_indices,
                              const GpuDeviceArrayStruct<const T*>& input_ptrs,
                              T* output) {
  const int32 output_size = first_dim_size * slice_size;
  auto config = GetGpuLaunchConfig(output_size, gpu_device);

  TF_CHECK_OK(GpuLaunchKernel(TfraDynamicStitchKernel<T>, config.block_count,
                              config.thread_per_block, 0, gpu_device.stream(),
                              slice_size, output_size, input_indices,
                              input_ptrs, output));
}

template <typename T>
void TfraDynamicStitchFastGPUImpl(OpKernelContext* c, const int32 slice_size,
                                  const OpInputList& indices_inputs,
                                  const OpInputList& data_inputs, T* output) {
  int32 ninner = 0;
  int32 nsegments = indices_inputs.size();

  GpuDeviceArrayOnHost<int32> indices_vec_sizes(c, nsegments);
  OP_REQUIRES_OK(c, indices_vec_sizes.Init());
  for (int i = 0; i < nsegments; ++i) {
    int64_t ele_num = indices_inputs[i].NumElements();
    ninner = max(static_cast<int64_t>(ninner), ele_num);
    indices_vec_sizes.Set(i, ele_num);
  }
  OP_REQUIRES_OK(c, indices_vec_sizes.Finalize());

  GpuDeviceArrayOnHost<const int32*> indices_inputs_base(c, nsegments);
  OP_REQUIRES_OK(c, indices_inputs_base.Init());
  for (int i = 0; i < nsegments; ++i) {
    indices_inputs_base.Set(i, indices_inputs[i].flat<int32>().data());
  }
  OP_REQUIRES_OK(c, indices_inputs_base.Finalize());

  GpuDeviceArrayOnHost<const T*> data_inputs_base(c, data_inputs.size());
  OP_REQUIRES_OK(c, data_inputs_base.Init());
  for (int i = 0; i < data_inputs.size(); ++i) {
    data_inputs_base.Set(i, data_inputs[i].template flat<T>().data());
  }
  OP_REQUIRES_OK(c, data_inputs_base.Finalize());

  auto& gpu_device = c->eigen_gpu_device();
  Gpu2DLaunchConfig config = GetGpu2DLaunchConfig(
      ninner, nsegments, gpu_device, TfraDynamicStitchFastKernel<T>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  TF_CHECK_OK(GpuLaunchKernel(
      TfraDynamicStitchFastKernel<T>, config.block_count,
      config.thread_per_block, 0, gpu_device.stream(), slice_size, nsegments,
      indices_vec_sizes.data(), indices_inputs_base.data(),
      data_inputs_base.data(), output));
}

#define REGISTER_GPU(T)                                           \
  template void TfraDynamicStitchGPUImpl(                         \
      const Eigen::GpuDevice& gpu_device, const int32 slice_size, \
      const int32 first_dim_size,                                 \
      const GpuDeviceArrayStruct<int32>& input_indices,           \
      const GpuDeviceArrayStruct<const T*>& input_ptrs, T* output);

TF_CALL_bool(REGISTER_GPU);
TF_CALL_int8(REGISTER_GPU);
TF_CALL_int32(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);

#undef REGISTER_GPU

#define REGISTER_GPU(T)                                                  \
  template void TfraDynamicStitchFastGPUImpl(                            \
      OpKernelContext* c, const int32 slice_size,                        \
      const OpInputList& indices_inputs, const OpInputList& data_inputs, \
      T* output);

TF_CALL_bool(REGISTER_GPU);
TF_CALL_int8(REGISTER_GPU);
TF_CALL_int32(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);

#undef REGISTER_GPU

}  // namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
