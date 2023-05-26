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

#ifndef TFRA_CORE_KERNELS_SEGMENT_REDUCTION_OPS_IMPL_H_
#define TFRA_CORE_KERNELS_SEGMENT_REDUCTION_OPS_IMPL_H_

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <stdio.h>
#include <unistd.h>

#include <vector>

#include "segment_reduction_ops.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/segment_reduction_ops.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/util.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA
#if TF_VERSION_INTEGER >= 2070  // 2.7.0
#include "tensorflow/core/util/gpu_solvers.h"
#elif TF_VERSION_INTEGER >= 2040  // 2.4.0
#include "tensorflow/core/util/cuda_solvers.h"
#else
#include "tensorflow/core/kernels/cuda_solvers.h"
#endif                          // TF_VERSION_INTEGER >= 2040
#if TF_VERSION_INTEGER >= 2110  // 2.11.0
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_activation.h"
#else
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#endif
using stream_executor::cuda::ScopedActivateExecutorContext;
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/rocm.h"
#include "tensorflow/core/util/cuda_solvers.h"
using stream_executor::rocm::ScopedActivateExecutorContext;
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <class Device, class T, class Tindex, bool has_num_segments>
class SparseSegmentSumGpuOp : public AsyncOpKernel {
 public:
  explicit SparseSegmentSumGpuOp(OpKernelConstruction* context)
      : AsyncOpKernel(context){};

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& input_data = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);

    OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(indices.shape()),
                      errors::InvalidArgument("indices should be a vector."),
                      done);
    const int64 num_indices = indices.NumElements();

    OP_REQUIRES_ASYNC(
        context, TensorShapeUtils::IsVector(segment_ids.shape()),
        errors::InvalidArgument("segment_ids should be a vector."), done);

    OP_REQUIRES_ASYNC(
        context, num_indices == segment_ids.NumElements(),
        errors::InvalidArgument("indices and segment_ids should have"
                                "same length."),
        done);

    ScratchSpace<Tindex> output_rows_host(context, /*size=*/1,
                                          /*on_host=*/true);
    auto stream = context->op_device_context()->stream();

    if (has_num_segments) {
      const Tensor& num_segments = context->input(3);
      CHECK(output_rows_host.tensor().CopyFrom(num_segments,
                                               num_segments.shape()));
    } else {
      se::DeviceMemoryBase last_segment_id_on_device(const_cast<Tindex*>(
          segment_ids.template flat<Tindex>().data() + num_indices - 1));
      OP_REQUIRES_ASYNC(
          context,
          stream
              ->ThenMemcpy(output_rows_host.mutable_data(),
                           last_segment_id_on_device, sizeof(Tindex))
              .ok(),
          errors::Internal(
              "SparseSegmentSumGpuOp: failed to copy output_rows to host."),
          done);
    }

    const Tindex input_dims = input_data.dims();
    OP_REQUIRES_ASYNC(
        context, input_dims >= 1,
        errors::InvalidArgument("indices and segment_ids should have "
                                "same length."),
        done);

    Tindex element_size = 1;
    const TensorShape input_shape = input_data.shape();
    if (input_dims > 1) {
      for (Tindex i = 1; i < input_dims; i++) {
        element_size *= input_shape.dim_size(i);
      }
    }

    OP_REQUIRES_OK_ASYNC(context, stream->BlockHostUntilDone(), done);
    Tindex output_rows = *output_rows_host.data();
    // Since segment_ids counts from 0 for output position, the output_rows
    // is increased by 1, if there is no specified num_segments value.
    if (!has_num_segments) {
      output_rows++;
    }
    OP_REQUIRES_ASYNC(context, output_rows > 0,
                      errors::InvalidArgument("Segment ids must be >= 0"),
                      done);
    TensorShape output_shape = input_data.shape();
    output_shape.set_dim(0, output_rows);

    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(0, output_shape, &output), done);

    functor::SparseSegmentSumFunctor<T, Tindex> executant(
        output_rows, num_indices, num_indices * element_size, input_data,
        indices, segment_ids, output);

    ScopedActivateExecutorContext scoped_activation{stream->parent()};
    executant(context, context->eigen_device<GPUDevice>());

#if TF_VERSION_INTEGER >= 2090  // 2.9.0
    context->device()
        ->tensorflow_accelerator_device_info()
        ->event_mgr->ThenExecute(stream, done);
#else
    context->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
        stream, done);
#endif
  }
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_SEGMENT_REDUCTION_OPS_IMPL_H_
