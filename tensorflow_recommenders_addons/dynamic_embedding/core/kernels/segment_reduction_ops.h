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

#ifndef TFRA_CORE_KERNELS_SEGMENT_REDUCTION_OPS_H_
#define TFRA_CORE_KERNELS_SEGMENT_REDUCTION_OPS_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#if TF_VERSION_INTEGER >= 2160
#include "unsupported/Eigen/CXX11/Tensor"
#else
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#endif
namespace tensorflow {

class OpKernelContext;

namespace functor {

typedef Eigen::GpuDevice GPUDevice;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename T, typename Index>
struct SparseSegmentSumFunctor {
  const Index output_rows;
  const Index num_indices;
  const Index data_size;
  const Tensor& input;
  const Tensor& indices;
  const Tensor& segment_ids;
  Tensor* output;

  explicit SparseSegmentSumFunctor(const Index output_rows,
                                   const Index num_indices,
                                   const Index data_size,
                                   const Tensor& input_data,
                                   const Tensor& indices,
                                   const Tensor& segment_ids, Tensor* output)
      : output_rows(output_rows),
        num_indices(num_indices),
        data_size(data_size),
        input(input_data),
        indices(indices),
        segment_ids(segment_ids),
        output(output) {}

  void operator()(OpKernelContext* ctx, const GPUDevice& d);
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace functor

}  // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_SEGMENT_REDUCTION_OPS_H_
