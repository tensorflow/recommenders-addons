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

#include "sparse_fill_empty_rows_op.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

template <typename Device, typename T>
class SparseFillEmptyRowsOp : public OpKernel {
 public:
  explicit SparseFillEmptyRowsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    functor::SparseFillEmptyRowsFunctor<Device, T>()(context);
  }
};

#if GOOGLE_CUDA
#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("TfraSparseFillEmptyRows") \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<type>("T"), \
                          SparseFillEmptyRowsOp<GPUDevice, type>)
TF_CALL_int8(REGISTER_KERNELS);
TF_CALL_int32(REGISTER_KERNELS);
TF_CALL_half(REGISTER_KERNELS);
TF_CALL_float(REGISTER_KERNELS);
#undef REGISTER_KERNELS
#endif  // GOOGLE_CUDA
}  // namespace tensorflow
