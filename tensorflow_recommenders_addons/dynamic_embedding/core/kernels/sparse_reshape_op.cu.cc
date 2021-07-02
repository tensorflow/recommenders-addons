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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparse_reshape_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

template <class IndexType>
__global__ void SparseReshapeKernel(const IndexType* input_indices_in,
                                    const int nnz, const int input_dim,
                                    const int output_dim,
                                    const IndexType* input_shape,
                                    const IndexType* target_shape,
                                    IndexType* output_indices,
                                    IndexType* output_shape) {
  int64 dense_size = 1;
  for (int d = 0; d < input_dim; ++d) {
    dense_size *= input_shape[d];
  }
  int64 product = 1;
  int unknown_index = -1;
  for (int d = 0; d < output_dim; ++d) {
    const int64 size = target_shape[d];
    if (size == -1) {
      unknown_index = d;
      output_shape[d] = 1;
    } else {
      product *= size;
      output_shape[d] = size;
    }
  }

  if (unknown_index != -1) {
    const int64 missing = dense_size / product;
    output_shape[unknown_index] = missing;
  }

#define RESHAPE_KERNEL_MAX_DIM 32
  int64 input_strides[RESHAPE_KERNEL_MAX_DIM];
  int64 output_strides[RESHAPE_KERNEL_MAX_DIM];
#undef RESHAPE_KERNEL_MAX_DIM
  // compute input strides
  input_strides[input_dim - 1] = 1;
  for (int i = input_dim - 2; i >= 0; i--) {
    input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
  }
  // compute output strides
  output_strides[output_dim - 1] = 1;
  for (int i = output_dim - 2; i >= 0; i--) {
    output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
  }

  GPU_1D_KERNEL_LOOP(idx, nnz) {
    IndexType id = 0;
#pragma unroll
    for (int i = 0; i < input_dim; i++) {
      id = id + input_strides[i] * input_indices_in[idx * input_dim + i];
    }
#pragma unroll
    for (int i = 0; i < output_dim; i++) {
      output_indices[idx * output_dim + i] = id / output_strides[i];
      id = id % output_strides[i];
    }
  }
}

namespace functor {

template class SparseReshapeFunctor<int>;
template class SparseReshapeFunctor<int64>;

void SparseReshapeFunctor<GPUDevice>::operator()(OpKernelContext* context) {
  auto input_indices = context->input(0);
  auto input_shape = context->input(1);
  auto target_shape = context->input(2);

  OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices.shape()),
              errors::InvalidArgument(
                  "Input indices should be a matrix but received shape ",
                  input_indices.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape.shape()),
              errors::InvalidArgument(
                  "Input shape should be a vector but received shape ",
                  input_shape.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsVector(target_shape.shape()),
              errors::InvalidArgument(
                  "Target shape should be a vector but received shape ",
                  target_shape.shape().DebugString()));

  const int nnz = input_indices.shape().dim_size(0);
  const int rank = input_shape.NumElements();
  const int output_rank = target_shape.NumElements();

  Tensor* output_indices = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, TensorShape({nnz, output_rank}),
                                          &output_indices));

  Tensor* output_shape = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(
                              1, TensorShape({output_rank}), &output_shape));

  auto d = context->eigen_gpu_device();
  GpuLaunchConfig config = GetGpuLaunchConfig(nnz, d);
  TF_CHECK_OK(GpuLaunchKernel(
      SparseReshapeKernel<int64>, config.block_count, config.thread_per_block,
      0, d.stream(), input_indices.flat<int64>().data(), nnz, rank, output_rank,
      input_shape.flat<int64>().data(), target_shape.flat<int64>().data(),
      output_indices->flat<int64>().data(),
      output_shape->flat<int64>().data()));
}
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
