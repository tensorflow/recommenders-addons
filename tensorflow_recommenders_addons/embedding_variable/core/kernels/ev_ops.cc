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
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow_recommenders_addons/embedding_variable/core/kernels/embedding_var.h"
#include "tensorflow_recommenders_addons/embedding_variable/core/kernels/ev_op_helpers.h"

namespace tensorflow {
namespace ev {

#define REGISTER_EV_HANDLE(ktype, vtype)                        \
  REGISTER_KERNEL_BUILDER(Name("EVHandleOp")                    \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<ktype>("Tkey")    \
                              .TypeConstraint<vtype>("Tvalue"), \
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
                              .TypeConstraint<ktype>("Tkey")    \
                              .TypeConstraint<vtype>("Tvalue"), \
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
    OP_REQUIRES_OK(c, c->GetAttr("Tvalue", &dtype_));
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
    const Tensor& invalid_key = context->input(2);
    OP_REQUIRES_OK(context,
                   LookupOrCreateResource<EmbeddingVar<TKey, TValue>>(
                       context, HandleFromInput(context, 0), &embedding_var,
                       [this, default_values,
                        invalid_key](EmbeddingVar<TKey, TValue>** ptr) {
                         *ptr = new EmbeddingVar<TKey, TValue>("EmbeddingVar");
                         return (*ptr)->Init(default_values, invalid_key);
                       }));
    core::ScopedUnref unref_me(embedding_var);
    embedding_var->SetInitialized();
  }

 private:
  DataType dtype_;
  TensorShape shape_;
};

#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("InitializeEVOp")                \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<ktype>("Tkey")    \
                              .TypeConstraint<vtype>("Tvalue"), \
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
#define REGISTER_KERNELS(ktype, vtype)                         \
  REGISTER_KERNEL_BUILDER(Name("EVIsInitializedOp")            \
                              .TypeConstraint<ktype>("Tkey")   \
                              .TypeConstraint<vtype>("Tvalue") \
                              .Device(DEVICE_CPU),             \
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

#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("EVGather")                      \
                              .Device(DEVICE_CPU)               \
                              .HostMemory("resource")           \
                              .HostMemory("indices")            \
                              .HostMemory("default_value")      \
                              .HostMemory("output")             \
                              .TypeConstraint<ktype>("Tkey")    \
                              .TypeConstraint<vtype>("Tvalue"), \
                          EVGatherOp<ktype, vtype>)

#define REGISTER_CPU_KERNELS(type) \
  REGISTER_KERNELS(int32, type);   \
  REGISTER_KERNELS(int64, type)

REGISTER_CPU_KERNELS(float);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename TKey, typename TValue, typename TStep>
class EVSparseApplyGradientDescentOp : public OpKernel {
 public:
  explicit EVSparseApplyGradientDescentOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) {
    auto locks = MaybeLockEmbeddingVariableInputMutexesInOrder<TKey, TValue>(
        ctx, use_exclusive_lock_, {0});

    EmbeddingVar<TKey, TValue>* embedding_var = nullptr;
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
      auto indices_vec = indices.vec<TKey>();
      TValue lr_scalar = lr.scalar<TValue>()();
      TStep global_step_scalar = global_step.scalar<TStep>()();

      if (inner_dim > 0) {
        auto grad_flat = grad.flat_outer_dims<TValue>();

        for (int64 i = 0; i < N; i++) {
          const TKey index = indices_vec(i);
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

#define REGISTER_KERNELS(ktype, vtype, stype) \
  REGISTER_KERNEL_BUILDER(                    \
      Name("EVSparseApplyGradientDescent")    \
          .Device(DEVICE_CPU)                 \
          .HostMemory("var")                  \
          .TypeConstraint<ktype>("Tkey")      \
          .TypeConstraint<vtype>("Tvalue")    \
          .TypeConstraint<stype>("Tstep"),    \
      EVSparseApplyGradientDescentOp<ktype, vtype, stype>);

#define REGISTER_CPU_KERNELS(T)      \
  REGISTER_KERNELS(int32, T, int32); \
  REGISTER_KERNELS(int32, T, int64); \
  REGISTER_KERNELS(int64, T, int32); \
  REGISTER_KERNELS(int64, T, int64);

REGISTER_CPU_KERNELS(float);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename TKey, typename TValue, typename TStep>
class EVSparseApplyAdagradOp : public OpKernel {
 public:
  explicit EVSparseApplyAdagradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) {
    auto locks = MaybeLockEmbeddingVariableInputMutexesInOrder<TKey, TValue>(
        ctx, use_exclusive_lock_, {0, 1});

    EmbeddingVar<TKey, TValue>* var = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));

    EmbeddingVar<TKey, TValue>* accum = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &accum));

    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    const Tensor& global_step = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(global_step.shape()),
                errors::InvalidArgument("global_step is not a scalar: ",
                                        global_step.shape().DebugString()));

    int64 inner_dim = 1;
    TensorShape var_shape({var->Size(), var->ValueLen()});
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }

    const TKey N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      if (inner_dim > 0) {
        auto indices_vec = indices.vec<TKey>();
        auto grad_flat = grad.flat_outer_dims<TValue>();
        TValue lr_scalar = lr.scalar<TValue>()();
        TStep gs = global_step.scalar<TStep>()();

        for (int64 i = 0; i < N; i++) {
          const TKey index = indices_vec(i);

          auto a = accum->flat(index, gs);
          auto g = grad_flat.template chip<0>(i);
          auto v = var->flat(index, gs);

          a += g.square();
          v -= g.constant(lr_scalar) * g * a.rsqrt();
        }
      }
    }
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(ktype, vtype, stype)                  \
  REGISTER_KERNEL_BUILDER(Name("EVSparseApplyAdagrad")         \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<ktype>("Tkey")   \
                              .TypeConstraint<vtype>("Tvalue") \
                              .TypeConstraint<stype>("Tstep"), \
                          EVSparseApplyAdagradOp<ktype, vtype, stype>);
#define REGISTER_CPU_KERNELS(T)      \
  REGISTER_KERNELS(int32, T, int32); \
  REGISTER_KERNELS(int32, T, int64); \
  REGISTER_KERNELS(int64, T, int32); \
  REGISTER_KERNELS(int64, T, int64);

REGISTER_CPU_KERNELS(float);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename TKey, typename TValue, typename TStep>
class EVSparseApplyAdamOp : public OpKernel {
 public:
  explicit EVSparseApplyAdamOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) {
    auto locks = MaybeLockEmbeddingVariableInputMutexesInOrder<TKey, TValue>(
        ctx, use_exclusive_lock_, {0, 1, 2});
    EmbeddingVar<TKey, TValue>* var = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &var));

    EmbeddingVar<TKey, TValue>* m = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &m));

    EmbeddingVar<TKey, TValue>* v = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &v));

    const Tensor& beta1_power = ctx->input(3);
    const Tensor& beta2_power = ctx->input(4);
    const Tensor& lr = ctx->input(5);
    const Tensor& beta1 = ctx->input(6);
    const Tensor& beta2 = ctx->input(7);
    const Tensor& epsilon = ctx->input(8);
    const Tensor& grad = ctx->input(9);
    const Tensor& indices = ctx->input(10);
    const Tensor& global_step = ctx->input(11);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    int64 inner_dim = 1;
    TensorShape var_shape({var->Size(), var->ValueLen()});
    for (int d = 1; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(global_step.shape()),
                errors::InvalidArgument("global_step is not a scalar: ",
                                        global_step.shape().DebugString()));

    const int64 N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    if (N > 0) {
      TValue beta1_power_scalar = beta1_power.scalar<TValue>()();
      TValue beta2_power_scalar = beta2_power.scalar<TValue>()();
      TValue lr_scalar = lr.scalar<TValue>()();
      TValue beta1_scalar = beta1.scalar<TValue>()();
      TValue beta2_scalar = beta2.scalar<TValue>()();
      TValue epsilon_scalar = epsilon.scalar<TValue>()();
      const TValue alpha =
          lr_scalar *
          Eigen::numext::sqrt(static_cast<TValue>(1) - beta2_power_scalar) /
          (static_cast<TValue>(1) - beta1_power_scalar);

      auto DoWork = [this, ctx, inner_dim, &var, &m, &v, &grad, &indices,
                     &beta1_power_scalar, &beta2_power_scalar, &lr_scalar,
                     &beta1_scalar, &beta2_scalar, &epsilon_scalar, &alpha,
                     &global_step](int64 start_i, int64 limit_i) {
        if (inner_dim > 0) {
          auto grad_flat = grad.flat_outer_dims<TValue>();
          auto indices_vec = indices.vec<TKey>();

          TStep gs = global_step.scalar<TStep>()();

          for (int64 i = static_cast<int64>(start_i);
               i < static_cast<int64>(limit_i); i++) {
            const TKey index = indices_vec(i);

            auto var_i = var->flat(index, gs);
            auto m_a = m->flat(index, gs);
            auto v_a = v->flat(index, gs);

            auto g = grad_flat.template chip<0>(i);
            m_a += (g - m_a) * (static_cast<TValue>(1) - beta1_scalar);
            v_a += (g.square() - v_a) * (static_cast<TValue>(1) - beta2_scalar);
            var_i -= (m_a * alpha) / (v_a.sqrt() + epsilon_scalar);
          }
        }
      };

      const int64 cost = 1000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            DoWork);
    }
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(ktype, vtype, stype)                  \
  REGISTER_KERNEL_BUILDER(Name("EVSparseApplyAdam")            \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<ktype>("Tkey")   \
                              .TypeConstraint<vtype>("Tvalue") \
                              .TypeConstraint<stype>("Tstep"), \
                          EVSparseApplyAdamOp<ktype, vtype, stype>);
#define REGISTER_CPU_KERNELS(T)      \
  REGISTER_KERNELS(int32, T, int32); \
  REGISTER_KERNELS(int32, T, int64); \
  REGISTER_KERNELS(int64, T, int32); \
  REGISTER_KERNELS(int64, T, int64);

REGISTER_CPU_KERNELS(float);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class EVExportOp : public OpKernel {
 public:
  explicit EVExportOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    std::vector<TKey> key_list;
    std::vector<TValue*> valueptr_list;
    int64 total_size = ev->GetSnapshot(&key_list, &valueptr_list);

    Tensor* key = nullptr;
    Tensor* val = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({total_size}), &key));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({total_size, ev->ValueLen()}),
                                  &val));
    auto key_flat = key->flat<TKey>();
    auto val_matrix = val->matrix<TValue>();
    for (int64_t i = 0; i < total_size; ++i) {
      key_flat(i) = key_list[i];
      TValue* value = valueptr_list[i];
      Eigen::array<Eigen::DenseIndex, 1> dims({ev->ValueLen()});
      typename TTypes<TValue>::Flat value_flat =
          typename TTypes<TValue>::Flat(value, dims);
      for (int64 j = 0; j < ev->ValueLen(); ++j) {
        val_matrix(i, j) = value_flat(j);
      }
    }
  }
};

#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("EVExport")                      \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<ktype>("Tkey")    \
                              .TypeConstraint<vtype>("Tvalue"), \
                          EVExportOp<ktype, vtype>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(int32, T);   \
  REGISTER_KERNELS(int64, T);

REGISTER_CPU_KERNELS(float);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename TKey, typename TValue>
class EVImportOp : public OpKernel {
 public:
  explicit EVImportOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) {
    EmbeddingVar<TKey, TValue>* ev = nullptr;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &ev));
    Tensor key = ctx->input(1);
    Tensor val = ctx->input(2);
    auto key_flat = key.flat<TKey>();
    auto val_matrix = val.matrix<TValue>();
    for (int64_t i = 0; i < key.NumElements(); ++i) {
      auto value = &val_matrix(i, 0);
      ev->LookupOrCreate(key_flat(i), value);
    }
  }
};

#define REGISTER_KERNELS(ktype, vtype)                          \
  REGISTER_KERNEL_BUILDER(Name("EVImport")                      \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<ktype>("Tkey")    \
                              .TypeConstraint<vtype>("Tvalue"), \
                          EVImportOp<ktype, vtype>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(int32, T);   \
  REGISTER_KERNELS(int64, T);

REGISTER_CPU_KERNELS(float);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

}  // namespace ev
}  // namespace tensorflow
