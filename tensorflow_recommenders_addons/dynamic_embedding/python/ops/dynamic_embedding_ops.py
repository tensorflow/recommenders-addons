# Copyright 2020 The TensorFlow Recommenders-Addons Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# lint-as: python3
"""
Dynamic Embedding is designed for Large-scale Sparse Weights Training.
See [Sparse Domain Isolation](https://github.com/tensorflow/community/pull/237)
"""

from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.shadow_embedding_ops import DEResourceVariable
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.embedding_weights import EmbeddingWeights

from tensorflow.python.eager import tape as tape_record
if not hasattr(tape_record, 'record_operation'):
  # tf version >= 2.13.0
  from tensorflow.python.eager import record as tape_record
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
try:  # tf version >= 2.14.0
  from tensorflow.python.framework.tensor import Tensor
except:
  from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
try:  # tf version >= 2.14.0
  from tensorflow.python.ops.array_ops_stack import stack
except:
  from tensorflow.python.ops.array_ops import stack
try:  # tf version >= 2.10.0
  from tensorflow.python.trackable import base as trackable
except:
  from tensorflow.python.training.tracking import base as trackable
try:  # The data_structures has been moved to the new package in tf 2.11
  from tensorflow.python.trackable import data_structures
except:
  from tensorflow.python.training.tracking import data_structures

try:  # tf version >= 2.14.0
  from tensorflow.python.distribute import distribute_lib as distribute_ctx
  assert hasattr(distribute_ctx, 'has_strategy')
except:
  from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx


def embedding_lookup_unique(params,
                            ids,
                            partition_strategy=None,
                            name=None,
                            validate_indices=None,
                            max_norm=None,
                            return_trainable=False):
  """Version of embedding_lookup that avoids duplicate lookups.
  This can save communication in the case of repeated ids.
  Same interface as embedding_lookup.

  Args:
    params: A dynamic_embedding.Variable instance.
    ids: a tensor with any shape as same dtype of params.key_dtype.
    partition_strategy: No used, for API compatiblity with `nn.emedding_lookup`.
    name: A name for the operation. Name is optional in graph mode and required
      in eager mode.
    validate_indices: No used, just for compatible with nn.embedding_lookup .
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value.
    return_trainable: optional, If True, also return TrainableWrapper

  Returns:
    A tensor with shape [shape of ids] + [dim],
      dim is equal to the value dim of params.
      containing the values from the params tensor(s) for keys in ids.
    trainable_wrap:
      A TrainableWrapper object used to fill the Optimizers `var_list`
        Only provided if `return_trainable` is True.
  """
  with ops.name_scope(name, "EmbeddingLookupUnique", [params, ids]):
    ids = ops.convert_to_tensor(ids)
    shape = array_ops.shape(ids)
    ids_flat = array_ops.reshape(ids, math_ops.reduce_prod(shape,
                                                           keepdims=True))
    unique_ids, idx = array_ops.unique(ids_flat)
    unique_embeddings, trainable_ = de.embedding_lookup(
        params,
        unique_ids,
        partition_strategy=partition_strategy,
        name=name,
        validate_indices=None,
        max_norm=validate_indices,
        return_trainable=True)
    embeddings_flat = array_ops.gather(unique_embeddings, idx)
    embeddings_shape = array_ops.concat(
        [shape, array_ops.shape(unique_embeddings)[1:]], 0)
    embeddings = array_ops.reshape(embeddings_flat, embeddings_shape)
    embeddings.set_shape(ids.get_shape().concatenate(
        unique_embeddings.get_shape()[1:]))
    return (embeddings, trainable_) if return_trainable else embeddings


def embedding_lookup_sparse(
    params: EmbeddingWeights,
    sp_ids,
    sp_weights,
    partition_strategy=None,  # no used
    name="embedding_lookup_sparse",
    combiner="mean",
    max_norm=None,
    return_trainable=False,
):
  """Provides a dynamic version of embedding_lookup_sparse
      similar with tf.nn.embedding_lookup_sparse.

    This op assumes that there is at least one id for each row in the dense tensor
    represented by sp_ids (i.e. there are no rows with empty features), and that
    all the indices of sp_ids are in canonical row-major order.

    It also assumes that all id values lie in the range [0, p0), where p0
    is the sum of the size of params along dimension 0.

    Args:
      params: A single `IEmbeddingVariable`, `dynamic_embedding.Variable` instance representing
        the complete embedding tensor and a new TrainableWrapper will be created and return
         or a `ShadowVariable` / `HvdVariable` instance, then params will be return without creating a new TrainableWrapper
      sp_ids: N x M `SparseTensor` of int64 ids where N is typically batch size
        and M is arbitrary.
      sp_weights: either a `SparseTensor` of float / double weights, or `None` to
        indicate all weights should be taken to be 1. If specified, `sp_weights`
        must have exactly the same shape and indices as `sp_ids`.
      partition_strategy: No used.
      name: a name for the operation. Name is optional in graph mode and required
        in eager mode.
      combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
        and "sum" are supported. "sum" computes the weighted sum of the embedding
        results for each row. "mean" is the weighted sum divided by the total
        weight. "sqrtn" is the weighted sum divided by the square root of the sum
        of the squares of the weights.
      max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
        than this value, before combining.
      return_trainable: optional, If True, also return TrainableWrapper create by
        `dynamic_embedding.embedding_lookup`

    Returns:
      combined_embeddings: A dense tensor representing the combined embeddings
        for the sparse ids. For each row in the dense tensor represented by
        `sp_ids`, the op looks up the embeddings for all ids in that row,
        multiplies them by the corresponding weight, and combines these embeddings
        as specified.

        In other words, if

          `shape(combined params) = [+infinity, dim]`

        and

          `shape(sp_ids) = shape(sp_weights) = [d0, d1, ..., dn]`

        then

          `shape(output) = [d0, dim]`.

        For instance, if params dim=20, and sp_ids / sp_weights are

          ```python
          [0, 0]: id 1, weight 2.0
          [0, 1]: id 3, weight 0.5
          [1, 0]: id 0, weight 1.0
          [2, 3]: id 1, weight 3.0
          ```

        with `combiner`="mean", then the output will be a 3x20 matrix where

          ```python
          output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
          output[1, :] = (params[0, :] * 1.0) / 1.0
          output[2, :] = (params[1, :] * 3.0) / 3.0
          ```
      trainable_wrap:
        A TrainableWrapper object used to fill the Optimizers `var_list`
          Only provided if `return_trainable` is True.
    Raises:
      TypeError: If `sp_ids` is not a `SparseTensor`, or if `sp_weights` is
        neither `None` nor `SparseTensor`.
      ValueError: If `combiner` is not one of {"mean", "sqrtn", "sum"}.
    """
  if combiner not in ("mean", "sqrtn", "sum"):
    raise ValueError("combiner must be one of 'mean', 'sqrtn' or 'sum'")

  if not isinstance(sp_ids, sparse_tensor.SparseTensor):
    raise TypeError("sp_ids must be SparseTensor")

  ignore_weights = sp_weights is None
  if not ignore_weights:
    if not isinstance(sp_weights, sparse_tensor.SparseTensor):
      raise TypeError("sp_weights must be either None or SparseTensor")

  scope = variable_scope.get_variable_scope()
  full_name = scope.name + "/" + name if scope.name else name
  with ops.name_scope(full_name + "/"):
    segment_ids = sp_ids.indices[:, 0]
    if segment_ids.dtype != dtypes.int32:
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)

    ids = sp_ids.values
    ids, idx = array_ops.unique(ids)
    embeddings, trainable_ = params.embedding_lookup(ids,
                                                     name=name,
                                                     max_norm=max_norm)

    if embeddings.dtype in (dtypes.float16, dtypes.bfloat16):
      embeddings = math_ops.cast(embeddings, dtypes.float32)
    if not ignore_weights:
      weights = sp_weights.values
      if weights.dtype != embeddings.dtype:
        weights = math_ops.cast(weights, embeddings.dtype)

      embeddings = array_ops.gather(embeddings, idx)

      # Reshape weights to allow broadcast
      ones = array_ops.fill(
          array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0), 1)
      bcast_weights_shape = array_ops.concat([array_ops.shape(weights), ones],
                                             0)

      orig_weights_shape = weights.get_shape()
      weights = array_ops.reshape(weights, bcast_weights_shape)

      # Set the weight shape, since after reshaping to bcast_weights_shape,
      # the shape becomes None.
      if embeddings.get_shape().ndims is not None:
        weights.set_shape(
            orig_weights_shape.concatenate(
                [1 for _ in range(embeddings.get_shape().ndims - 1)]))

      embeddings *= weights

      if combiner == "sum":
        embeddings = math_ops.segment_sum(embeddings, segment_ids, name=name)
      elif combiner == "mean":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
        weight_sum = math_ops.segment_sum(weights, segment_ids)
        embeddings = math_ops.divide(embeddings, weight_sum, name=name)
      elif combiner == "sqrtn":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
        weights_squared = math_ops.pow(weights, 2)
        weight_sum = math_ops.segment_sum(weights_squared, segment_ids)
        weight_sum_sqrt = math_ops.sqrt(weight_sum)
        embeddings = math_ops.divide(embeddings, weight_sum_sqrt, name=name)
      else:
        assert False, "Unrecognized combiner"
    else:
      assert idx is not None
      if combiner == "sum":
        embeddings = de.math.sparse_segment_sum(embeddings,
                                                idx,
                                                segment_ids,
                                                name=name)
      elif combiner == "mean":
        embeddings = math_ops.sparse_segment_mean(embeddings,
                                                  idx,
                                                  segment_ids,
                                                  name=name)
      elif combiner == "sqrtn":
        embeddings = math_ops.sparse_segment_sqrt_n(embeddings,
                                                    idx,
                                                    segment_ids,
                                                    name=name)
      else:
        assert False, "Unrecognized combiner"

    return (embeddings, trainable_) if return_trainable else embeddings


def safe_embedding_lookup_sparse(
    embedding_weights: EmbeddingWeights,
    sparse_ids,
    sparse_weights=None,
    combiner="mean",
    default_id=None,
    name="safe_embedding_lookup_sparse",
    partition_strategy=None,  # no used
    max_norm=None,
    return_trainable=False,
):
  """Provides a dynamic version of `tf.nn.safe_embedding_lookup_sparse`.

    Lookup embedding results, accounting for empty features and invalid weights.

    Any IDs will be treated as valid include non-positive IDs.
    Invalid weights (<= 0) are pruned from input weights, as well as any IDs
    with non-positive weight. For an entry with no features, the embedding vector
    for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

    The ids and weights may be multi-dimensional. Embeddings are always aggregated
    along the last dimension.

    Args:
      embedding_weights: A single `IEmbeddingVariable`, either `dynamic_embedding.Variable` or `ShadowVariable`
       or `HvdVariable` instance representing the complete embedding tensor.
      sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
        ids. `d_0` is typically batch size.
      sparse_weights: `SparseTensor` of same shape as `sparse_ids`, containing
        float weights corresponding to `sparse_ids`, or `None` if all weights are
        be assumed to be 1.0.
      combiner: A string specifying how to combine embedding results for each
        entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean" the
        default.
      default_id: The id to use for an entry with no features.
      name: A name for this operation. Name is optional in graph mode and required
        in eager mode.
      partition_strategy: A string specifying the partitioning strategy. Currently
        `"div"` and `"mod"` are supported. Default is `"div"`.
      max_norm: If not `None`, all embeddings are l2-normalized to max_norm before
        combining.

    Returns:
      combined_embeddings:
        A dense `Tensor` of shape `[d_0, d_1, ..., d_{n-1}, e_1, ..., e_m]`.
      trainable_wrap:
        A TrainableWrapper object used to fill the Optimizers `var_list`
          Only provided if `return_trainable` is True.

    Raises:
      ValueError: if `embedding_weights` is empty.
  """
  embedding_weights.verify_embedding_weights(sparse_ids, sparse_weights)

  scope = variable_scope.get_variable_scope()
  full_name = scope.name + "/" + name if scope.name else name

  with ops.name_scope(full_name + "/"):
    # Reshape higher-rank sparse ids and weights to linear segment ids.
    original_shape = sparse_ids.dense_shape
    original_rank_dim = tensor_shape.dimension_value(
        sparse_ids.dense_shape.get_shape()[0])
    original_rank = (array_ops.size(original_shape)
                     if original_rank_dim is None else original_rank_dim)
    sparse_ids = de.math.sparse_reshape(
        sparse_ids,
        [
            math_ops.reduce_prod(
                array_ops.slice(original_shape, [0], [original_rank - 1])),
            array_ops.gather(original_shape, original_rank - 1),
        ],
    )
    if sparse_weights is not None:
      sparse_weights = sparse_tensor.SparseTensor(sparse_ids.indices,
                                                  sparse_weights.values,
                                                  sparse_ids.dense_shape)

    # Prune invalid weights.
    if combiner != "sum":
      sparse_ids, sparse_weights = _prune_invalid_weights(
          sparse_ids, sparse_weights)

    # Fill in dummy values for empty features, if necessary.
    sparse_ids, is_row_empty = de.math.sparse_fill_empty_rows(
        sparse_ids, default_id or 0)
    if sparse_weights is not None:
      sparse_weights, _ = de.math.sparse_fill_empty_rows(sparse_weights, 1.0)

    result, trainable_ = embedding_lookup_sparse(
        embedding_weights,
        sparse_ids,
        sparse_weights,
        combiner=combiner,
        partition_strategy=partition_strategy,
        name=name + "/embedding_lookup_sparse",
        max_norm=max_norm,
        return_trainable=True,
    )

    if default_id is None:
      # Broadcast is_row_empty to the same shape as embedding_lookup_result,
      # for use in Select.
      is_row_empty = array_ops.tile(
          array_ops.reshape(is_row_empty, [-1, 1]),
          stack([1, array_ops.shape(result)[1]]),
      )

      result = array_ops.where(is_row_empty,
                               array_ops.zeros_like(result),
                               result,
                               name="where")

    # Reshape back from linear ids back into higher-dimensional dense result.
    final_result = array_ops.reshape(
        result,
        array_ops.concat(
            [
                array_ops.slice(
                    math_ops.cast(original_shape, dtypes.int32),
                    [0],
                    [original_rank - 1],
                ),
                array_ops.slice(array_ops.shape(result), [1], [-1]),
            ],
            0,
        ),
    )
    final_result.set_shape(
        tensor_shape.unknown_shape(
            (tensor_shape.Dimension(original_rank_dim) - 1).value).concatenate(
                result.get_shape()[1:]))
    return (final_result, trainable_) if return_trainable else final_result


def _prune_invalid_weights(sparse_ids, sparse_weights):
  """Prune invalid weights (< 0) from the input ids and weights."""
  if sparse_weights is not None:
    is_weights_valid = math_ops.greater(sparse_weights.values, 0)
    sparse_ids = sparse_ops.sparse_retain(sparse_ids, is_weights_valid)
    sparse_weights = sparse_ops.sparse_retain(sparse_weights, is_weights_valid)
  return sparse_ids, sparse_weights


def get_model_mode():
  """ get model mode.

  Returns:
    A string: `train` or 'inference'
  """
  return de.ModelMode.CURRENT_SETTING


def enable_train_mode():
  """ enable train mode.
  """
  de.ModelMode.CURRENT_SETTING = de.ModelMode.TRAIN


def enable_inference_mode():
  """ set inference mode.
  """
  de.ModelMode.CURRENT_SETTING = de.ModelMode.INFERENCE


def trainable_wrapper_filter(iterable_object_in,
                             test_unaggregated_function=None):
  dense_grads_and_vars_aggregated_out = []
  sparse_grads_and_vars_unaggregated_out = []
  if test_unaggregated_function is None:
    test_unaggregated_function = lambda x: isinstance(
        x, de.TrainableWrapper) or isinstance(x, DEResourceVariable)
  for item in iterable_object_in:
    if test_unaggregated_function(item):
      sparse_grads_and_vars_unaggregated_out.append(item)
    else:
      dense_grads_and_vars_aggregated_out.append(item)
  return tuple(dense_grads_and_vars_aggregated_out), tuple(
      sparse_grads_and_vars_unaggregated_out)
