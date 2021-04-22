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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "segment_reduction_ops.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

template <typename T, typename Index, int OuterDimTileSize>
__global__ void SortedSparseSegmentSumCustomKernel(
    const Index input_outer_dim_size, const Index inner_dim_size,
    const Index output_outer_dim_size, const Index* indices,
    const Index* segment_ids, const T* input, T* output,
    const Index total_stripe_count) {
  for (int stripe_index : GpuGridRangeX(total_stripe_count)) {
    const Index segment_offset = stripe_index % inner_dim_size;
    const Index input_outer_dim_index_base =
        stripe_index / inner_dim_size * Index(OuterDimTileSize);

    T sum = T(0);
    Index first_segment_id = segment_ids[input_outer_dim_index_base];
    Index last_output_segment_id = output_outer_dim_size;

    const Index actual_stripe_height =
        min(Index(OuterDimTileSize),
            input_outer_dim_size - input_outer_dim_index_base);
    for (Index j = 0; j < actual_stripe_height; j++) {
      Index current_output_segment_id =
          segment_ids[input_outer_dim_index_base + j];
      Index current_input_index_id = indices[input_outer_dim_index_base + j];

      if (current_output_segment_id > last_output_segment_id) {
        const Index output_index =
            last_output_segment_id * inner_dim_size + segment_offset;
        if (last_output_segment_id == first_segment_id) {
          GpuAtomicAdd(output + output_index, sum);
        } else {
          *(output + output_index) = sum;
        }
        sum = T(0);
      }
      sum +=
          ldg(input + current_input_index_id * inner_dim_size + segment_offset);
      last_output_segment_id = current_output_segment_id;
    }

    const Index output_index =
        last_output_segment_id * inner_dim_size + segment_offset;
    GpuAtomicAdd(output + output_index, sum);
  }
}

namespace functor {

// TODO(Lifann): enable indices validation on GPU. Right now
// checking for indicies out of bound in the kernel would
// require copying code between GPU/CPU, and thus slow.
template <typename T, typename Index>
void SparseSegmentSumFunctor<T, Index>::operator()(OpKernelContext* ctx,
                                                   const GPUDevice& d) {
  auto stream = ctx->op_device_context()->stream();
  auto output_flat = output->flat_outer_dims<T>();
  auto data_ptr = input.template flat<T>().data();
  const auto indices_flat = indices.flat<Index>();
  const auto segment_flat = segment_ids.flat<Index>();

  if (output_flat.size() == 0) {
    return;
  }

  GpuLaunchConfig config = GetGpuLaunchConfig(output_flat.size(), d);
  TF_CHECK_OK(GpuLaunchKernel(SetZero<T>, config.block_count,
                              config.thread_per_block, 0, d.stream(),
                              output_flat.size(), output_flat.data()));
  if (data_size == 0 || segment_ids.shape().num_elements() == 0) {
    return;
  }

  const Index input_total_size = data_size;
  const Index input_outer_dim_size = segment_flat.dimension(0);
  const Index input_inner_dim_size = input_total_size / input_outer_dim_size;
  const int OuterDimTileSize = 8;

  const Index input_outer_dim_num_stripe =
      Eigen::divup(input_outer_dim_size, Index(OuterDimTileSize));

  const Index total_stripe_count =
      input_inner_dim_size * input_outer_dim_num_stripe;

  if (total_stripe_count <= 0) return;
  config = GetGpuLaunchConfig(total_stripe_count, d);
  TF_CHECK_OK(GpuLaunchKernel(
      SortedSparseSegmentSumCustomKernel<T, Index, OuterDimTileSize>,
      config.block_count, config.thread_per_block, 0, d.stream(),
      input_outer_dim_size, input_inner_dim_size, output_rows,
      indices_flat.data(), segment_flat.data(), data_ptr, output_flat.data(),
      total_stripe_count));
}

#define DEFINE_SPARSE_SEGMENT_SUM_GPU_SPECS_INDEX(T, Index) \
  template struct SparseSegmentSumFunctor<T, Index>

#define DEFINE_SPARSE_SEGMENT_SUM_GPU_SPECS(T)         \
  DEFINE_SPARSE_SEGMENT_SUM_GPU_SPECS_INDEX(T, int32); \
  DEFINE_SPARSE_SEGMENT_SUM_GPU_SPECS_INDEX(T, int64);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_SPARSE_SEGMENT_SUM_GPU_SPECS);

#undef DEFINE_SPARSE_SEGMENT_SUM_GPU_SPECS
#undef DEFINE_SPARSE_SEGMENT_SUM_GPU_SPECS_INDEX

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
