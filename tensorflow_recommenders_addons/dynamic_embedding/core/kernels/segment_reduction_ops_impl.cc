/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/math_ops.cc.
#include "segment_reduction_ops_impl.h"

namespace tensorflow {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_SORTED_KERNELS(type, index_type)                        \
  REGISTER_KERNEL_BUILDER(Name("TFRA>SparseSegmentSum")                      \
                              .Device(DEVICE_GPU)                            \
                              .TypeConstraint<type>("T")                     \
                              .TypeConstraint<index_type>("Tidx"),           \
                          SparseSegmentSumGpuOp<GPUDevice, type, index_type, \
                                                /*has_num_segments=*/false>)

#define REGISTER_GPU_SORTED_KERNELS_ALL(type) \
  REGISTER_GPU_SORTED_KERNELS(type, int32);   \
  REGISTER_GPU_SORTED_KERNELS(type, int64);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_SORTED_KERNELS_ALL);

#undef REGISTER_GPU_SORTED_KERNELS
#undef REGISTER_GPU_SORTED_KERNELS_ALL

#define REGISTER_GPU_SORTED_KERNELS(type, index_type)                        \
  REGISTER_KERNEL_BUILDER(Name("TFRA>SparseSegmentSumWithNumSegments")       \
                              .Device(DEVICE_GPU)                            \
                              .HostMemory("num_segments")                    \
                              .TypeConstraint<type>("T")                     \
                              .TypeConstraint<index_type>("Tidx"),           \
                          SparseSegmentSumGpuOp<GPUDevice, type, index_type, \
                                                /*has_num_segments=*/true>)

#define REGISTER_GPU_SORTED_KERNELS_ALL(type) \
  REGISTER_GPU_SORTED_KERNELS(type, int32);   \
  REGISTER_GPU_SORTED_KERNELS(type, int64);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_SORTED_KERNELS_ALL);

#undef REGISTER_GPU_SORTED_KERNELS
#undef REGISTER_GPU_SORTED_KERNELS_ALL

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
