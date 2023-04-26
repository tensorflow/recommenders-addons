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

Status DynamicStitchShapeFunction(InferenceContext* c) {
  int32_t num_partitions;
  TF_RETURN_IF_ERROR(c->GetAttr("N", &num_partitions));

  bool all_indices_constant = true;
  int32_t max_index = -1;
  ShapeHandle extra_shape = c->UnknownShape();
  for (int i = 0; i < num_partitions; ++i) {
    const Tensor* indices_t = c->input_tensor(i);
    if (indices_t == nullptr) {
      all_indices_constant = false;
    }

    ShapeHandle indices_shape = c->input(i);
    ShapeHandle data_shape = c->input(i + num_partitions);
    if (!c->RankKnown(indices_shape)) {
      continue;
    }
    const int64_t indices_rank = c->Rank(indices_shape);

    // Assert that data_shape starts with indices_shape.
    ShapeHandle unused;
    TF_RETURN_IF_ERROR(
        c->MergePrefix(data_shape, indices_shape, &unused, &unused));

    // The rest belongs to output.
    ShapeHandle rest;
    TF_RETURN_IF_ERROR(c->Subshape(data_shape, indices_rank, &rest));
    TF_RETURN_IF_ERROR(c->Merge(extra_shape, rest, &extra_shape));

    if (indices_t != nullptr) {
      // The length is based on the highest index from flattened indices.
      const int32* indices = indices_t->flat<int32>().data();
      int64_t count = indices_t->NumElements();
      for (int64_t i = 0; i < count; ++i) {
        if (indices[i] > max_index) {
          max_index = indices[i];
        }
      }
    }
  }

  ShapeHandle output_shape = c->Vector(
      all_indices_constant ? c->MakeDim(max_index + 1) : c->UnknownDim());
  TF_RETURN_IF_ERROR(c->Concatenate(output_shape, extra_shape, &output_shape));
  c->set_output(0, output_shape);
  return TFOkStatus;
}

}  // namespace

#if GOOGLE_CUDA

REGISTER_OP(PREFIX_OP_NAME(DynamicPartition))
    .Input("data: T")
    .Input("partitions: int32")
    .Output("outputs: num_partitions * T")
    .Attr("num_partitions: int")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      int64_t num_partitions;
      TF_RETURN_IF_ERROR(c->GetAttr("num_partitions", &num_partitions));

      ShapeHandle data_shape = c->input(0);
      ShapeHandle partitions_shape = c->input(1);

      if (!c->RankKnown(partitions_shape)) {
        return shape_inference::UnknownShape(c);
      }

      const int64_t rank = c->Rank(partitions_shape);

      // data shape must start with partitions_shape
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(
          c->MergePrefix(data_shape, partitions_shape, &unused, &unused));

      // The partition shape is dynamic in the 0th dimension, and matches
      // data_shape in the remaining dimensions.
      ShapeHandle unknown_dim0 = c->MakeShape({c->UnknownDim()});

      ShapeHandle data_suffix_shape;
      TF_RETURN_IF_ERROR(c->Subshape(data_shape, rank, &data_suffix_shape));
      ShapeHandle result_shape;
      TF_RETURN_IF_ERROR(
          c->Concatenate(unknown_dim0, data_suffix_shape, &result_shape));

      for (int i = 0; i < c->num_outputs(); ++i) {
        c->set_output(i, result_shape);
      }

      return TFOkStatus;
    });

REGISTER_OP(PREFIX_OP_NAME(DynamicStitch))
    .Input("indices: N * int32")
    .Input("data: N * T")
    .Output("merged: T")
    .Attr("N : int >= 1")
    .Attr("T : type")
    .SetShapeFn(DynamicStitchShapeFunction);

REGISTER_OP(PREFIX_OP_NAME(DynamicStitchFast))
    .Input("indices: N * int32")
    .Input("data: N * T")
    .Output("merged: T")
    .Attr("N : int >= 1")
    .Attr("T : type")
    .SetShapeFn(DynamicStitchShapeFunction);

REGISTER_OP(PREFIX_OP_NAME(ParallelDynamicStitch))
    .Input("indices: N * int32")
    .Input("data: N * T")
    .Output("merged: T")
    .Attr("N : int >= 1")
    .Attr("T : type")
    .SetShapeFn(DynamicStitchShapeFunction);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
