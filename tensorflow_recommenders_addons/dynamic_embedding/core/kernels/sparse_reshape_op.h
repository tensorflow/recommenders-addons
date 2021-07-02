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

#ifndef TFRA_CORE_KERNELS_SPARSE_RESHAPE_OP_H_
#define TFRA_CORE_KERNELS_SPARSE_RESHAPE_OP_H_

// Functor definition for SparseReshapeOp, must be compilable by nvcc.

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace functor {

template <typename Device>
struct SparseReshapeFunctor {
  void operator()(OpKernelContext* context);
};

#if GOOGLE_CUDA
template <>
struct SparseReshapeFunctor<Eigen::GpuDevice> {
  void operator()(OpKernelContext* context);
};
#endif  // GOOGLE_CUDA

}  // namespace functor
}  // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_SPARSE_RESHAPE_OP_H_
