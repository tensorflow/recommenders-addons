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

#include "sparsehash/dense_hash_map"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_recommenders_addons/embedding_variable/core/kernels/embedding_var.h"
#include "tensorflow_recommenders_addons/embedding_variable/core/kernels/ev_op_helpers.h"

namespace tensorflow {
namespace ev {

#define REGISTER_EV_HANDLE(ktype, vtype)                       \
  REGISTER_KERNEL_BUILDER(Name("EVHandleOp")                   \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<ktype>("Tkeys")  \
                              .TypeConstraint<vtype>("dtype"), \
                          ResourceHandleOp<EmbeddingVar<ktype, vtype>>);
REGISTER_EV_HANDLE(int32, float)
REGISTER_EV_HANDLE(int64, float)
#undef REGISTER_EV_HANDLE

template <typename T, typename TKey, typename TValue>
class EVShapeOp : public OpKernel {
 public:
  explicit EVShapeOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    EmbeddingVar<TKey, TValue>* embedding_var = nullptr;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &embedding_var));
    core::ScopedUnref unref_me(embedding_var);
    TensorShape shape({embedding_var->Size(), embedding_var->ValueLen()});
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {shape.dims()}, &output));
    for (int i = 0; i < shape.dims(); ++i) {
      output->flat<T>()(i) = shape.dim_size(i);
    }
  }
};

#define REGISTER_KV_VARIABLE_SHAPE(type, ktype, vtype)          \
  REGISTER_KERNEL_BUILDER(Name("EVShape")                       \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<type>("out_type") \
                              .TypeConstraint<ktype>("Tkeys"),  \
                          EVShapeOp<type, ktype, vtype>);
REGISTER_KV_VARIABLE_SHAPE(int32, int32, float)
REGISTER_KV_VARIABLE_SHAPE(int32, int64, float)
REGISTER_KV_VARIABLE_SHAPE(int64, int32, float)
REGISTER_KV_VARIABLE_SHAPE(int64, int64, float)
#undef REGISTER_KV_VARIABLE_SHAPE

template <typename TKey, typename TValue>
class InitializeEVOp : public OpKernel {
 public:
  explicit InitializeEVOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape_));
    OP_REQUIRES(c, shape_.dims() == 1,
                errors::InvalidArgument("EV dimension must be 1"));
  }

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context, dtype_ == context->input(1).dtype(),
                errors::InvalidArgument(
                    "Variable and value dtypes don't match; respectively, ",
                    dtype_, " and ", context->input(1).dtype()));
    EmbeddingVar<TKey, TValue>* embedding_var = nullptr;
    const Tensor& default_values = context->input(1);
    OP_REQUIRES_OK(
        context, LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
                     context, HandleFromInput(context, 0), &embedding_var,
                     [this, default_values](EmbeddingVar<TKey, TValue>** ptr) {
                       *ptr = new EmbeddingVar<TKey, TValue>("EmbeddingVar");
                       return (*ptr)->Init(default_values);
                     }));
    core::ScopedUnref unref_me(embedding_var);
    embedding_var->SetInitialized();
  }

 private:
  DataType dtype_;
  TensorShape shape_;
};

#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("InitializeEVOp")               \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<ktype>("Tkeys")  \
                              .TypeConstraint<vtype>("dtype"), \
                          InitializeEVOp<ktype, vtype>);
#define REGISTER_KERNELS_ALL_INDEX(type) \
  REGISTER_KERNELS(int32, type)          \
  REGISTER_KERNELS(int64, type)
REGISTER_KERNELS_ALL_INDEX(float)
#undef REGISTER_KERNELS_ALL_INDEX
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class EVIsInitializedOp : public OpKernel {
 public:
  explicit EVIsInitializedOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &output));
    EmbeddingVar<TKey, TValue>* embedding_var = nullptr;
    bool found;
    if (LookupResource<EmbeddingVar<TKey, TValue>>(ctx, HandleFromInput(ctx, 0),
                                                   &embedding_var)
            .ok()) {
      found = embedding_var->IsInitialized();
      embedding_var->Unref();
    } else {
      found = false;
    }

    output->flat<bool>()(0) = found;
  }
};
#define REGISTER_KERNELS(ktype, vtype)                        \
  REGISTER_KERNEL_BUILDER(Name("EVIsInitializedOp")           \
                              .TypeConstraint<ktype>("Tkeys") \
                              .Device(DEVICE_CPU),            \
                          EVIsInitializedOp<ktype, vtype>);
REGISTER_KERNELS(int32, float)
REGISTER_KERNELS(int64, float)
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class EVGatherOp : public OpKernel {
 public:
  explicit EVGatherOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    EmbeddingVar<TKey, TValue>* embedding_var = nullptr;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &embedding_var));
    int64 ev_dim_size = embedding_var->ValueLen();

    const Tensor& indices = context->input(1);
    const int64 N = indices.NumElements();

    Tensor default_values(context->input(2));
    auto default_values_matrix = default_values.shaped<TValue, 2>(
        {default_values.NumElements() / ev_dim_size, ev_dim_size});

    TensorShape result_shape = indices.shape();
    TensorShape value_shape({ev_dim_size});
    result_shape.AppendShape(value_shape);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, result_shape, &out));

    if (N > 0) {
      auto out_flat = out->shaped<TValue, 2>({N, out->NumElements() / N});
      TValue* out_base = &out_flat(0, 0);

      auto indices_flat = indices.flat<TKey>();
      const int64 indices_size = static_cast<int64>(indices_flat.dimension(0));
      const int64 slice_elems = out_flat.dimension(1);
      OP_REQUIRES(
          context, ev_dim_size == slice_elems,
          errors::InvalidArgument(
              "hashmap's value_len should same with output's dimension(1)",
              std::to_string(slice_elems), std::to_string(ev_dim_size)));

      const size_t slice_bytes = slice_elems * sizeof(TValue);
      for (int64 i = 0; i < indices_size; i++) {
        TValue* default_v = &default_values_matrix(i, 0);
        TValue* mem_val =
            embedding_var->LookupOrCreate(indices_flat(i), default_v);
        memcpy(out_base + i * slice_elems, mem_val, slice_bytes);
      }
    }
  }
};

#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("EVGather")                     \
                              .Device(DEVICE_CPU)              \
                              .HostMemory("resource")          \
                              .HostMemory("indices")           \
                              .HostMemory("default_value")     \
                              .HostMemory("output")            \
                              .TypeConstraint<vtype>("dtype")  \
                              .TypeConstraint<ktype>("Tkeys"), \
                          EVGatherOp<ktype, vtype>)

#define REGISTER_CPU_KERNELS(type) \
  REGISTER_KERNELS(int32, type);   \
  REGISTER_KERNELS(int64, type)

REGISTER_CPU_KERNELS(float);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename T, typename Tindex, typename Tstep>
class EVSparseApplyGradientDescentOp : public OpKernel {
 public:
  explicit EVSparseApplyGradientDescentOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) {
    auto locks = MaybeLockEmbeddingVariableInputMutexesInOrder<Tindex, T>(
        ctx, use_exclusive_lock_, {0});

    EmbeddingVar<Tindex, T>* embedding_var = nullptr;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &embedding_var));

    const Tensor& lr = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));

    const Tensor& grad = ctx->input(2);
    const Tensor& indices = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    const Tensor& global_step = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(global_step.shape()),
                errors::InvalidArgument("global_step is not a scalar: ",
                                        global_step.shape().DebugString()));

    int64 inner_dim = 1;
    TensorShape var_shape({embedding_var->Size(), embedding_var->ValueLen()});
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const int64 N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    if (N > 0) {
      auto indices_vec = indices.vec<Tindex>();
      T lr_scalar = lr.scalar<T>()();
      Tstep global_step_scalar = global_step.scalar<Tstep>()();

      if (inner_dim > 0) {
        auto grad_flat = grad.flat_outer_dims<T>();

        for (int64 i = 0; i < N; i++) {
          const Tindex index = indices_vec(i);
          auto g = grad_flat.template chip<0>(i);
          auto v = embedding_var->flat(index, global_step_scalar);
          v -= g.constant(lr_scalar) * g;
        }
      }
    }
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices, Tstep)                        \
  REGISTER_KERNEL_BUILDER(Name("EVSparseApplyGradientDescent")      \
                              .Device(DEVICE_CPU)                   \
                              .HostMemory("var")                    \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<Tindices>("Tindices") \
                              .TypeConstraint<Tstep>("Tstep"),      \
                          EVSparseApplyGradientDescentOp<T, Tindices, Tstep>);

#define REGISTER_CPU_KERNELS(T)      \
  REGISTER_KERNELS(T, int64, int32); \
  REGISTER_KERNELS(T, int64, int64);

REGISTER_CPU_KERNELS(float);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

}  // namespace ev
}  // namespace tensorflow
