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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/kernels/sparse_fill_empty_rows_op.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/lib/nvhash/cub/cub/device/device_scan.cuh"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

// calculate how many rows are empty and record their location
__global__ void SparseFillEmptyRowCountKernel(
    const int64* indices, const int nnz, const int64* input_shape,
    int* row_nnz_count,       // size: num_rows
    int64* input_row_offset,  // size: num_rows + 1
    int64* output_row_offset  // size: num_rows + 1
) {
  GPU_1D_KERNEL_LOOP(idx, nnz) {
    int64 _row = indices[idx * 2];
    atomicAdd(row_nnz_count + _row, 1);
  }
}

__global__ void SparseFillEmptyRowAddOneKernel(const int64* input_shape,
                                               int* row_nnz_count) {
  const int64 num_rows = input_shape[0];
  GPU_1D_KERNEL_LOOP(id_row, num_rows) {
    if (row_nnz_count[id_row] == 0) {
      row_nnz_count[id_row] += 1;
    }
  }
}

// copy the original data to output data address and fill default value to empty
// rows
template <class T>
__global__ void SparseFillEmptyRowFillKernel(
    // inputs
    const int64* input_indices, const T* input_values, const int64* input_shape,
    const T* default_value, const int64* input_row_offset,
    const int64* output_row_offset,
    // outputs
    int64* output_indices, T* output_values, bool* empty_row_indicator,
    int64* reverse_index_map) {
  const int64 num_rows = input_shape[0];
  GPU_1D_KERNEL_LOOP(id_row, num_rows) {
#pragma unroll
    for (int i = 0; i < input_row_offset[id_row + 1] - input_row_offset[id_row];
         i++) {
      output_values[output_row_offset[id_row] + i] =
          input_values[input_row_offset[id_row] + i];
      output_indices[2 * (output_row_offset[id_row] + i) + 0] =
          id_row;  // no need to read indices from input again;
      output_indices[2 * (output_row_offset[id_row] + i) + 1] =
          input_indices[2 * (input_row_offset[id_row] + i) + 1];
      if (reverse_index_map) {
        reverse_index_map[input_row_offset[id_row] + i] =
            output_row_offset[id_row] + i;
      }
    }

    // for empty rows
    if (input_row_offset[id_row + 1] == input_row_offset[id_row]) {
      // insert default value
      output_values[output_row_offset[id_row]] = *default_value;
      output_indices[2 * output_row_offset[id_row] + 0] = id_row;
      output_indices[2 * output_row_offset[id_row] + 1] = 0;

      // mark as empty
      if (empty_row_indicator) {
        empty_row_indicator[id_row] = true;
      }
    }
  }
  return;
}

namespace functor {
template <typename T>
void SparseFillEmptyRowsGpuImpl(OpKernelContext* context,
                                const int64* input_indices,
                                const T* input_values, const int64 nnz,
                                const int64* input_shape,
                                const T* default_value) {
  auto d = context->eigen_gpu_device();
  auto OpStream = d.stream();
  int64 dense_row_number;

  // get the dense shape, which is stored in GPU.
  // If the dense shape is already in CPU, we don't need to do the copy here.
  cudaMemcpyAsync(&dense_row_number, input_shape, sizeof(int64),
                  cudaMemcpyDeviceToHost, OpStream);
  cudaStreamSynchronize(OpStream);

  // temp vector to store start index of each row
  Tensor input_row_offset;
  Tensor output_row_offset;
  Tensor row_nnz_count;  // temp buffer for the count kernel, count number of
                         // non-zero values on each row.

  // the size of input_row_offset and output_row_offset is dense_row_number+1,
  // because we need one extra place to store the initial value of the offset 0
  OP_REQUIRES_OK(context, context->allocate_temp(
                              DT_INT64, TensorShape({dense_row_number + 1}),
                              &input_row_offset));

  OP_REQUIRES_OK(context, context->allocate_temp(
                              DT_INT64, TensorShape({dense_row_number + 1}),
                              &output_row_offset));

  OP_REQUIRES_OK(
      context, context->allocate_temp(
                   // use DT_INT32 instead of DT_INT64, because CUDA atomic_add
                   // only support int32
                   DT_INT32, TensorShape({dense_row_number}), &row_nnz_count));

  cudaMemset(row_nnz_count.flat<int>().data(), 0,
             sizeof(int) * dense_row_number);
  cudaMemset(input_row_offset.flat<int64>().data(), 0, sizeof(int64));
  cudaMemset(output_row_offset.flat<int64>().data(), 0, sizeof(int64));

  // Get the number of rows in each row
  GpuLaunchConfig count_kernel_config = GetGpuLaunchConfig(nnz, d);
  TF_CHECK_OK(GpuLaunchKernel(
      SparseFillEmptyRowCountKernel, count_kernel_config.block_count,
      count_kernel_config.thread_per_block, 0, d.stream(), input_indices, nnz,
      input_shape, row_nnz_count.flat<int>().data(),
      input_row_offset.flat<int64>().data(),
      output_row_offset.flat<int64>().data()));

  /* Calculate the offset of each row of input
   *  example: the number of rows in each row: [3, 4, 0, 0, 6]
   *  the offset of each row of input: [0, 3, 7, 7, 7, 13]
   */
  // Determine temporary device storage requirements for inclusive prefix sum
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(
      NULL, temp_storage_bytes, row_nnz_count.flat<int>().data(),
      input_row_offset.flat<int64>().data() + 1, dense_row_number);

  // Allocate temporary storage for inclusive prefix sum
  Tensor temp_storage;
  OP_REQUIRES_OK(
      context,
      context->allocate_temp(
          DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
          &temp_storage));
  void* d_temp_storage = temp_storage.flat<int8>().data();

  // Run inclusive prefix sum
  cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_storage_bytes, row_nnz_count.flat<int>().data(),
      input_row_offset.flat<int64>().data() + 1, dense_row_number);

  /* Add 1 to the row whose row count is 0
   *  example: the number of rows in each row（row_nnz_count）: [3, 4, 0, 0, 6]
   *  row_nnz_count after the kernel: [3, 4, 1, 1, 6]
   */
  GpuLaunchConfig add_kernel_config = GetGpuLaunchConfig(nnz, d);
  TF_CHECK_OK(GpuLaunchKernel(
      SparseFillEmptyRowAddOneKernel, count_kernel_config.block_count,
      count_kernel_config.thread_per_block, 0, d.stream(), input_shape,
      row_nnz_count.flat<int>().data()));

  // Calculate the offset of each row of output
  cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_storage_bytes, row_nnz_count.flat<int>().data(),
      output_row_offset.flat<int64>().data() + 1, dense_row_number);

  // Read the output size from GPU, which is result of the first kernel.
  // copy nnz + num_of_empty_row = output_nnz to CPU
  int64 output_nnz;
  cudaMemcpyAsync(&output_nnz,
                  output_row_offset.flat<int64>().data() + dense_row_number,
                  sizeof(int64), cudaMemcpyDeviceToHost, OpStream);
  cudaStreamSynchronize(OpStream);

  // Allocate output tensors.
  Tensor* output_indices;
  Tensor* output_values;
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, TensorShape({output_nnz, 2}),
                                          &output_indices));
  OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({output_nnz}),
                                                   &output_values));

  bool* empty_row_indicator = nullptr;
  if (context->output_required(2)) {
    Tensor* empty_row_indicator_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, TensorShape({dense_row_number}),
                                            &empty_row_indicator_t));
    empty_row_indicator = empty_row_indicator_t->vec<bool>().data();
    // assume row not empty first
    cudaMemset(empty_row_indicator, false, sizeof(bool) * dense_row_number);
  }

  int64* reverse_index_map = nullptr;
  if (context->output_required(3)) {
    Tensor* reverse_index_map_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({nnz}),
                                                     &reverse_index_map_t));
    reverse_index_map = reverse_index_map_t->vec<int64>().data();
  }

  // Launch the second Kernel to move data and insert value to empty rows.
  GpuLaunchConfig config = GetGpuLaunchConfig(dense_row_number, d);
  TF_CHECK_OK(GpuLaunchKernel(
      SparseFillEmptyRowFillKernel<T>, config.block_count,
      config.thread_per_block, 0, d.stream(), input_indices, input_values,
      input_shape, default_value, input_row_offset.flat<int64>().data(),
      output_row_offset.flat<int64>().data(),
      output_indices->flat<int64>().data(), output_values->flat<T>().data(),
      empty_row_indicator, reverse_index_map));
}

template <typename T>
struct SparseFillEmptyRowsFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context) {
    auto input_indices = context->input(0);
    auto input_values = context->input(1);
    auto input_shape = context->input(2);
    auto default_value = context->input(3);

    const int64 nnz = input_indices.shape().dim_size(0);

    SparseFillEmptyRowsGpuImpl<T>(context, input_indices.flat<int64>().data(),
                                  input_values.flat<T>().data(), nnz,
                                  input_shape.flat<int64>().data(),
                                  default_value.flat<T>().data());
  }
};

#define DEFINE_GPU_KERNELS(type) \
  template struct SparseFillEmptyRowsFunctor<GPUDevice, type>;

TF_CALL_int8(DEFINE_GPU_KERNELS);
TF_CALL_int32(DEFINE_GPU_KERNELS);
TF_CALL_half(DEFINE_GPU_KERNELS);
TF_CALL_float(DEFINE_GPU_KERNELS);
TF_CALL_int64(DEFINE_GPU_KERNELS);
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
