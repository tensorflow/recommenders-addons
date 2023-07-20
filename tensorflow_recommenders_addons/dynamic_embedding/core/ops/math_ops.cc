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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/utils.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeAndType;
using shape_inference::ShapeHandle;

namespace {

Status SparseSegmentReductionShapeFn(InferenceContext* c) {
  ShapeHandle data_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &data_shape));

  ShapeHandle indices_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &indices_shape));

  ShapeHandle segment_ids_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &segment_ids_shape));

  // indices and segment_ids should merge cleanly.
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(indices_shape, segment_ids_shape, &unused));

  ShapeHandle subshape;
  TF_RETURN_IF_ERROR(c->Subshape(data_shape, 1, &subshape));

  ShapeHandle out;
  TF_RETURN_IF_ERROR(
      c->Concatenate(c->Vector(InferenceContext::kUnknownDim), subshape, &out));
  c->set_output(0, out);
  return TFOkStatus;
}

Status SparseSegmentReductionWithNumSegmentsShapeFn(InferenceContext* c) {
  ShapeHandle data_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &data_shape));

  ShapeHandle indices_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &indices_shape));

  ShapeHandle segment_ids_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &segment_ids_shape));

  ShapeHandle num_segments_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &num_segments_shape));

  // indices and segment_ids should merge cleanly.
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(indices_shape, segment_ids_shape, &unused));

  ShapeHandle subshape;
  TF_RETURN_IF_ERROR(c->Subshape(data_shape, 1, &subshape));

  ShapeHandle out;
  const Tensor* dim0 = c->input_tensor(3);
  if (dim0 == nullptr) {
    // We don't have the value at inference time, so the output
    // shape is unknown.
    TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(InferenceContext::kUnknownDim),
                                      subshape, &out));
  } else {
    auto dim0_value = dim0->scalar<int32>()();
    if (dim0_value < 0) {
      return errors::InvalidArgument(
          "Cannot specify a negative value for num_segments");
    }
    TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(dim0_value), subshape, &out));
  }
  c->set_output(0, out);
  return TFOkStatus;
}
}  // namespace

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER_OP(PREFIX_OP_NAME(SparseSegmentSum))
    .Input("data: T")
    .Input("indices: Tidx")
    .Input("segment_ids: Tsegmentids")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("Tsegmentids: {int32, int64} = DT_INT32")
    .SetShapeFn(SparseSegmentReductionShapeFn);

REGISTER_OP(PREFIX_OP_NAME(SparseSegmentSumWithNumSegments))
    .Input("data: T")
    .Input("indices: Tidx")
    .Input("segment_ids: Tsegmentids")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .Attr("Tsegmentids: {int32, int64} = DT_INT32")
    .SetShapeFn(SparseSegmentReductionWithNumSegmentsShapeFn);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA
REGISTER_OP("TfraSparseFillEmptyRows")
    .Input("indices: int64")
    .Input("values: T")
    .Input("dense_shape: int64")
    .Input("default_value: T")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Output("empty_row_indicator: bool")
    .Output("reverse_index_map: int64")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_indices = c->input(0);
      TF_RETURN_IF_ERROR(c->WithRank(input_indices, 2, &input_indices));
      ShapeHandle input_values = c->input(1);
      TF_RETURN_IF_ERROR(c->WithRank(input_values, 1, &input_values));
      ShapeHandle input_shape = c->input(2);
      TF_RETURN_IF_ERROR(c->WithRank(input_shape, 1, &input_shape));
      ShapeHandle default_value = c->input(3);
      TF_RETURN_IF_ERROR(c->WithRank(default_value, 0, &default_value));
      DimensionHandle N = c->Dim(input_indices, 0);
      TF_RETURN_IF_ERROR(c->Merge(N, c->Dim(input_values, 0), &N));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(input_indices, 1),
                                  c->Dim(input_shape, 0), &unused_dim));
      ShapeHandle output_indices =
          c->Matrix(InferenceContext::kUnknownDim, c->NumElements(input_shape));
      ShapeHandle output_values = c->Vector(InferenceContext::kUnknownDim);
      ShapeHandle constant_input_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &constant_input_shape));
      ShapeHandle empty_row_indicator =
          c->Vector(c->Dim(constant_input_shape, 0));
      ShapeHandle reverse_index_map = c->Vector(N);
      c->set_output(0, output_indices);
      c->set_output(1, output_values);
      c->set_output(2, empty_row_indicator);
      c->set_output(3, reverse_index_map);
      return TFOkStatus;
    });

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
