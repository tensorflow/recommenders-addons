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

#define EIGEN_USE_THREADS

#include "sparse_reshape_op.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

template <typename Device>
class SparseReshapeOp : public OpKernel {
 public:
  explicit SparseReshapeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    functor::SparseReshapeFunctor<Device>()(context);
  }
};

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("TfraSparseReshape").Device(DEVICE_GPU),
                        SparseReshapeOp<GPUDevice>);
#endif  // GOOGLE_CUDA
}  // namespace tensorflow
