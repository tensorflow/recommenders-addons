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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeAndType;
using ::tensorflow::shape_inference::ShapeHandle;

namespace tensorflow {
namespace ev {
namespace {

static ShapeHandle ShapeOrHandleShape(InferenceContext* c, int input) {
  auto* handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  return c->input(input);
}

Status ValidateVariableResourceHandle(InferenceContext* c,
                                      ShapeAndType* shape_and_type) {
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data == nullptr || handle_data->empty()) {
    shape_and_type->shape = c->UnknownShape();
    shape_and_type->dtype = DT_INVALID;
  } else {
    *shape_and_type = (*handle_data)[0];
    DataType value_dtype;
    TF_RETURN_IF_ERROR(c->GetAttr("dtype", &value_dtype));
    if (shape_and_type->dtype != value_dtype) {
      return errors::InvalidArgument(
          "Trying to read variable with wrong dtype. "
          "Expected ",
          DataTypeString(shape_and_type->dtype), " got ",
          DataTypeString(value_dtype));
    }
  }
  return Status::OK();
}

Status CreateAssignShapeFn(InferenceContext* c) {
  ShapeAndType handle_shape_and_type;
  TF_RETURN_IF_ERROR(ValidateVariableResourceHandle(c, &handle_shape_and_type));

  ShapeHandle value_shape = c->input(1);
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(
      c->Merge(handle_shape_and_type.shape, value_shape, &unused));
  return Status::OK();
}

static Status EVApplyGradientDescentShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
  ShapeHandle grad = ShapeOrHandleShape(c, 2);
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &indices));
  DimensionHandle unused2;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(grad, 0), &unused2));
  return Status::OK();
}

Status EVShapeShapeFn(InferenceContext* c) {
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data == nullptr || handle_data->empty()) {
    return errors::InvalidArgument("Handle doesn't have shape information.");
  }
  c->set_output(0, (*handle_data)[0].shape);
  return Status::OK();
}

}  // namespace

REGISTER_OP("EVHandleOp")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Attr("Tkeys: {int64}")
    .Output("resource: resource")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("dtype", &t));
      PartialTensorShape p;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &p));
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(p, &s));
      c->set_output_handle_shapes_and_types(0,
                                            std::vector<ShapeAndType>{{s, t}});

      return Status::OK();
    })
    .Doc(R"(
Creates a handle to a Embedding Variable resource.

container: the container this variable is placed in.
shared_name: the name by which this variable is referred to.
dtype: the type of this variable. Must agree with the dtypes
  of all ops using this variable.
shape: The (possibly partially specified) shape of this variable.
)");

REGISTER_OP("InitializeEVOp")
    .Input("resource: resource")
    .Input("value: dtype")
    .Input("empty_key: Tkeys")
    .Attr("Tkeys: {int32, int64}")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .SetShapeFn(CreateAssignShapeFn)
    .Doc(R"(
)");

REGISTER_OP("EVIsInitializedOp")
    .Input("resource: resource")
    .Output("is_initialized: bool")
    .Attr("Tkeys: {int32, int64}")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
)doc");

REGISTER_OP("EVShape")
    .Input("input: resource")
    .Output("output: out_type")
    .Attr("out_type: {int32, int64} = DT_INT32")
    .Attr("Tkeys: {int32, int64}")
    .SetShapeFn(EVShapeShapeFn)
    .Doc(R"doc(
)doc");

REGISTER_OP("EVGather")
    .Input("resource: resource")
    .Input("indices: Tkeys")
    .Input("default_value: dtype")
    .Attr("validate_indices: bool = true")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("Tkeys: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType handle_shape_and_type;
      TF_RETURN_IF_ERROR(
          ValidateVariableResourceHandle(c, &handle_shape_and_type));

      ShapeHandle unused;
      TF_RETURN_IF_ERROR(
          c->WithRankAtLeast(handle_shape_and_type.shape, 1, &unused));
      ShapeHandle params_subshape;
      params_subshape = handle_shape_and_type.shape;
      ShapeHandle indices_shape = c->input(1);
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, params_subshape, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
)doc");

REGISTER_OP("EVSparseApplyGradientDescent")
    .Input("var: resource")
    .Input("alpha: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Input("global_step: Tstep")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("Tstep: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(EVApplyGradientDescentShapeFn)
    .Doc(R"doc(
)doc");

// TODO(candy.dc): Other optimizer, such as: Adam, Adagrad

}  // namespace ev
}  // namespace tensorflow
