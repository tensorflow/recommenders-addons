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

/* After TensorFlow version 2.10.0, "Status::OK()" upgraded to "OkStatus()".
This code is for compatibility.*/
#if TF_VERSION_INTEGER >= 2100
#define TFOkStatus OkStatus()
#else
#define TFOkStatus Status::OK()
#endif

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
    TF_RETURN_IF_ERROR(c->GetAttr("Tvalue", &value_dtype));
    if (shape_and_type->dtype != value_dtype) {
      return errors::InvalidArgument(
          "Trying to read variable with wrong dtype. "
          "Expected ",
          DataTypeString(shape_and_type->dtype), " got ",
          DataTypeString(value_dtype));
    }
  }
  return TFOkStatus;
}

Status CreateAssignShapeFn(InferenceContext* c) {
  ShapeAndType handle_shape_and_type;
  TF_RETURN_IF_ERROR(ValidateVariableResourceHandle(c, &handle_shape_and_type));

  ShapeHandle value_shape = c->input(1);
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(
      c->Merge(handle_shape_and_type.shape, value_shape, &unused));
  return TFOkStatus;
}

static Status EVApplyGradientDescentShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
  ShapeHandle grad = ShapeOrHandleShape(c, 2);
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &indices));
  DimensionHandle unused2;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(grad, 0), &unused2));
  return TFOkStatus;
}

static Status EVApplyAdagradShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));

  ShapeHandle grad = ShapeOrHandleShape(c, 3);
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &indices));
  DimensionHandle unused2;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(grad, 0), &unused2));

  ShapeHandle grad_unknown_first;
  TF_RETURN_IF_ERROR(c->Subshape(grad, 1, &grad_unknown_first));
  return TFOkStatus;
}

static Status EVApplyAdamShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));       // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));       // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));       // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));       // epsilon

  ShapeHandle grad = ShapeOrHandleShape(c, 9);
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(10), 1, &indices));
  DimensionHandle unused2;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(grad, 0), &unused2));

  ShapeHandle grad_unknown_first;
  TF_RETURN_IF_ERROR(c->Subshape(grad, 1, &grad_unknown_first));
  return TFOkStatus;
}

Status EVShapeShapeFn(InferenceContext* c) {
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data == nullptr || handle_data->empty()) {
    return errors::InvalidArgument("Handle doesn't have shape information.");
  }
  c->set_output(0, (*handle_data)[0].shape);
  return TFOkStatus;
}

}  // namespace

REGISTER_OP("EVHandleOp")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("shape: shape")
    .Attr("Tkey: {int32, int64}")
    .Attr("Tvalue: type")
    .Output("resource: resource")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("Tvalue", &t));
      PartialTensorShape p;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &p));
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(p, &s));
      c->set_output_handle_shapes_and_types(0,
                                            std::vector<ShapeAndType>{{s, t}});

      return TFOkStatus;
    })
    .Doc(R"(
Creates a handle to a Embedding Variable resource.

container: the container this variable is placed in.
shared_name: the name by which this variable is referred to.
Tvalue: the type of this variable. Must agree with the dtypes
  of all ops using this variable.
shape: The (possibly partially specified) shape of this variable.
)");

REGISTER_OP("InitializeEVOp")
    .Input("resource: resource")
    .Input("value: Tvalue")
    .Input("empty_key: Tkey")
    .Attr("Tkey: {int32, int64}")
    .Attr("Tvalue: type")
    .Attr("shape: shape")
    .SetShapeFn(CreateAssignShapeFn)
    .Doc(R"(
)");

REGISTER_OP("EVIsInitializedOp")
    .Input("resource: resource")
    .Output("is_initialized: bool")
    .Attr("Tkey: {int32, int64}")
    .Attr("Tvalue: type")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
)doc");

REGISTER_OP("EVShape")
    .Input("input: resource")
    .Output("output: out_type")
    .Attr("out_type: {int32, int64} = DT_INT32")
    .Attr("Tkey: {int32, int64}")
    .Attr("Tvalue: type")
    .SetShapeFn(EVShapeShapeFn)
    .Doc(R"doc(
)doc");

REGISTER_OP("EVGather")
    .Input("resource: resource")
    .Input("indices: Tkey")
    .Input("default_value: Tvalue")
    .Output("output: Tvalue")
    .Attr("validate_indices: bool = true")
    .Attr("Tkey: {int32, int64}")
    .Attr("Tvalue: type")
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
      return TFOkStatus;
    })
    .Doc(R"doc(
)doc");

REGISTER_OP("EVSparseApplyGradientDescent")
    .Input("var: resource")
    .Input("alpha: Tvalue")
    .Input("grad: Tvalue")
    .Input("indices: Tkey")
    .Input("global_step: Tstep")
    .Attr("Tkey: {int32, int64}")
    .Attr("Tvalue: numbertype")
    .Attr("Tstep: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(EVApplyGradientDescentShapeFn)
    .Doc(R"doc(
)doc");

REGISTER_OP("EVSparseApplyAdagrad")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: Tvalue")
    .Input("grad: Tvalue")
    .Input("indices: Tkey")
    .Input("global_step: Tstep")
    .Attr("Tkey: {int32, int64}")
    .Attr("Tvalue: numbertype")
    .Attr("Tstep: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(EVApplyAdagradShapeFn)
    .Doc(R"doc(
)doc");

REGISTER_OP("EVSparseApplyAdam")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("beta1_power: Tvalue")
    .Input("beta2_power: Tvalue")
    .Input("lr: Tvalue")
    .Input("beta1: Tvalue")
    .Input("beta2: Tvalue")
    .Input("epsilon: Tvalue")
    .Input("grad: Tvalue")
    .Input("indices: Tkey")
    .Input("global_step: Tstep")
    .Attr("Tkey: {int32, int64}")
    .Attr("Tvalue: numbertype")
    .Attr("Tstep: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(EVApplyAdamShapeFn)
    .Doc(R"doc(
)doc");

REGISTER_OP("EVExport")
    .Input("ev: resource")
    .Output("keys: Tkey")
    .Output("values: Tvalue")
    .Attr("Tkey: {int32, int64}")
    .Attr("Tvalue: numbertype")
    .Doc(R"doc(
)doc");

REGISTER_OP("EVImport")
    .Input("ev: resource")
    .Input("keys: Tkey")
    .Input("values: Tvalue")
    .Attr("Tkey: {int32, int64}")
    .Attr("Tvalue: numbertype")
    .Doc(R"doc(
)doc");

}  // namespace ev
}  // namespace tensorflow

#ifdef TFOkStatus
#undef TFOkStatus
#endif