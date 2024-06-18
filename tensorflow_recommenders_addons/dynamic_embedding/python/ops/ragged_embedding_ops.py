import tensorflow as tf
from tensorflow.python.framework import dtypes, ops
from tensorflow.python.ops import resource_variable_ops, array_ops, math_ops, gen_ragged_array_ops, gen_math_ops
from tensorflow.python.ops.bincount_ops import validate_dense_weights
from tensorflow.python.ops.ragged import ragged_tensor, ragged_array_ops

from tensorflow_recommenders_addons.dynamic_embedding.python.ops.embedding_weights import EmbeddingWeights
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.shadow_embedding_ops import ShadowVariable


def _bincount(arr,
              weights=None,
              minlength=None,
              maxlength=None,
              dtype=dtypes.int32,
              name=None,
              axis=None,
              binary_output=False):

  name = "bincount" if name is None else name
  with tf.name_scope(name):
    arr = tf.convert_to_tensor(arr, name="arr")
    if weights is not None:
      weights = tf.convert_to_tensor(weights, name="weights")

    if weights is not None and binary_output:
      raise ValueError("Arguments `binary_output` and `weights` are mutually "
                       "exclusive. Please specify only one.")

    if not arr.dtype.is_integer:
      arr = math_ops.cast(arr, dtypes.int32)
    if axis is None:
      axis = 0

    if axis not in [0, -1]:
      raise ValueError(f"Unsupported value for argument axis={axis}. Only 0 and"
                       " -1 are currently supported.")

    array_is_nonempty = array_ops.size(arr) > 0
    output_size = math_ops.cast(array_is_nonempty,
                                arr.dtype) * (math_ops.reduce_max(arr) + 1)
    if minlength is not None:
      minlength = ops.convert_to_tensor(minlength,
                                        name="minlength",
                                        dtype=arr.dtype)
      output_size = gen_math_ops.maximum(minlength, output_size)
    if maxlength is not None:
      maxlength = ops.convert_to_tensor(maxlength,
                                        name="maxlength",
                                        dtype=arr.dtype)
      output_size = gen_math_ops.minimum(maxlength, output_size)

    if axis == 0:
      if weights is not None:
        weights = array_ops.reshape(weights, [-1])
      arr = array_ops.reshape(arr, [-1])

    weights = validate_dense_weights(arr, weights, dtype)
    return gen_math_ops.dense_bincount(input=arr,
                                       size=output_size,
                                       weights=weights,
                                       binary_output=binary_output)


# # for compatibility with tf 2.11
def _ragged_fill_empty_rows(value_rowids, values, nrows, default_value):
  # Convert default_value to the correct dtype
  default_value = tf.convert_to_tensor(default_value, dtype=values.dtype)

  # Determine the total number of rows and the maximum row index in value_rowids
  max_row_index = tf.reduce_max(value_rowids)
  total_rows = tf.maximum(nrows, max_row_index + 1)

  # Create a tensor of row lengths
  row_lengths = _bincount(value_rowids,
                          minlength=total_rows,
                          maxlength=total_rows,
                          dtype=value_rowids.dtype)

  # Identify empty rows
  empty_row_indicator = tf.equal(row_lengths, 0)

  # Generate default values for empty rows
  num_empty_rows = tf.reduce_sum(tf.cast(empty_row_indicator, tf.int32))
  default_values = tf.fill([num_empty_rows], default_value)

  # Create new value_rowids for empty rows
  empty_rows = tf.where(empty_row_indicator)
  new_value_rowids = tf.repeat(empty_rows, repeats=1)

  # Concatenate original and default values and row ids
  final_values = tf.concat([values, default_values], axis=0)
  final_value_rowids = tf.concat([value_rowids, new_value_rowids], axis=0)

  # Sort by rowids to maintain ragged tensor structure
  sorted_indices = tf.argsort(final_value_rowids)
  sorted_values = tf.gather(final_values, sorted_indices)
  sorted_value_rowids = tf.gather(final_value_rowids, sorted_indices)

  return sorted_value_rowids, sorted_values, empty_row_indicator


# # for compatibility with tf 2.11
def _fill_empty_rows(ragged_input, default_value, name=None):
  try:
    # if ragged_array_ops.fill_empty_rows is available, use it
    return ragged_array_ops.fill_empty_rows(ragged_input,
                                            default_value,
                                            name=name)
  except AttributeError:
    if not isinstance(ragged_input, tf.RaggedTensor):
      raise TypeError("ragged_input must be a RaggedTensor, got %s" %
                      type(ragged_input))
    default_value_tensor = tf.convert_to_tensor(default_value,
                                                dtype=ragged_input.dtype)

    output_value_rowids, output_values, empty_row_indicator = _ragged_fill_empty_rows(
        ragged_input.value_rowids(), ragged_input.values, ragged_input.nrows(),
        default_value_tensor)

    ragged_ordered_output = tf.RaggedTensor.from_value_rowids(
        values=output_values,
        value_rowids=output_value_rowids,
        nrows=ragged_input.nrows(),
        validate=False)
    return ragged_ordered_output, empty_row_indicator


def _embedding_lookup_sparse_impl(
    params,
    segment_ids,
    sp_weights,
    ids,
    combiner,
    ignore_weights,
    name,
):
  """Implementation of sparse embedding aggregation."""
  # Ensure we can query the devices below.
  segment_ids = ops.convert_to_tensor(segment_ids, name="segment_ids")

  ids, idx = array_ops.unique(ids)
  embeddings, _ = params.embedding_lookup(ids, name=name)
  if not ignore_weights:
    if segment_ids.dtype != dtypes.int32:
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)

    weights = sp_weights.values
    embeddings = array_ops.gather(embeddings, idx)

    original_dtype = embeddings.dtype
    if embeddings.dtype in (dtypes.float16, dtypes.bfloat16):
      # Cast low-precision embeddings to float32 during the computation to
      # avoid numerical issues.
      embeddings = math_ops.cast(embeddings, dtypes.float32)
    if weights.dtype != embeddings.dtype:
      weights = math_ops.cast(weights, embeddings.dtype)

    # Reshape weights to allow broadcast
    ones_shape = array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0)
    ones = array_ops.ones(ones_shape, dtype=dtypes.int32)
    bcast_weights_shape = array_ops.concat([array_ops.shape(weights), ones], 0)

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
      embeddings = math_ops.div_no_nan(embeddings, weight_sum, name=name)
    elif combiner == "sqrtn":
      embeddings = math_ops.segment_sum(embeddings, segment_ids)
      weights_squared = math_ops.pow(weights, 2)
      weight_sum = math_ops.segment_sum(weights_squared, segment_ids)
      weight_sum_sqrt = math_ops.sqrt(weight_sum)
      embeddings = math_ops.div_no_nan(embeddings, weight_sum_sqrt, name=name)
    else:
      assert False, "Unrecognized combiner"
    if embeddings.dtype != original_dtype:
      embeddings = math_ops.cast(embeddings, original_dtype)
  else:
    if segment_ids.dtype not in (dtypes.int32, dtypes.int64):
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)
    assert idx is not None
    if combiner == "sum":
      embeddings = math_ops.sparse_segment_sum(
          embeddings,
          idx,
          segment_ids,
          name=name,
      )
    elif combiner == "mean":
      embeddings = math_ops.sparse_segment_mean(
          embeddings,
          idx,
          segment_ids,
          name=name,
      )
    elif combiner == "sqrtn":
      embeddings = math_ops.sparse_segment_sqrt_n(
          embeddings,
          idx,
          segment_ids,
          name=name,
      )
    else:
      assert False, "Unrecognized combiner"

  return embeddings


def embedding_lookup_sparse(
    params: ShadowVariable,
    sp_ids: ragged_tensor.Ragged,
    sp_weights,
    name="embedding_lookup_sparse",
    combiner="mean",
):
  """Looks up embeddings for the given ids and weights from a list of tensors.

  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_ids (i.e. there are no rows with empty features), and that
  all the indices of sp_ids are in canonical row-major order.

  `sp_ids` and `sp_weights` (if not None) are `RaggedTensor`s with rank of 2.
  Embeddings are always aggregated along the last dimension.

  It also assumes that all id values lie in the range [0, p0), where p0
  is the sum of the size of params along dimension 0.

  Args:
    params: ShadowVariable
    sp_ids: `RaggedTensor` with rank 2. The rank is not verified for performance
      reasons.
    sparse_weights: `RaggedTensor` of same type and shape as `sparse_ids`,
      containing float / double weights corresponding to `sparse_ids`, or `None`
      if all weights are assumed to be 1.0.
    combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
      and "sum" are supported. "sum" computes the weighted sum of the embedding
      results for each row. "mean" is the weighted sum divided by the total
      weight. "sqrtn" is the weighted sum divided by the square root of the sum
      of the squares of the weights. Defaults to `mean`.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by `sp_ids`, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

    In other words, if

      `shape(combined params) = [p0, p1, ..., pm]`

    and

      `shape(sp_ids) = shape(sp_weights) = [d0, d1]`

    then

      `shape(output) = [d0, p1, ..., pm]`.

    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

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

  Raises:
    TypeError: If `sp_weights` is neither `None` nor of the same type as
      `sp_ids`.
    ValueError: If `combiner` is not one of {"mean", "sqrtn", "sum"}.
  """
  rt_ids = sp_ids
  rt_weights = sp_weights
  if combiner is None:
    combiner = "mean"
  if combiner not in ("mean", "sqrtn", "sum"):
    raise ValueError(
        f"combiner must be one of 'mean', 'sqrtn' or 'sum', got {combiner}")

  ignore_weights = rt_weights is None
  if not ignore_weights:
    if not isinstance(rt_weights, ragged_tensor.RaggedTensor):
      raise TypeError(f"sp_ids must be of the same type as sp_weights, "
                      f"received {{type(sp_ids).__name__!r}} for sp_ids and "
                      f"{{type(sp_weights).__name__!r}} for sp_weights.")
    rt_ids.values.get_shape().assert_is_compatible_with(
        rt_weights.values.get_shape())
    rt_ids.get_shape().assert_is_compatible_with(rt_weights.get_shape())

  with tf.name_scope(name or "embedding_lookup_sparse") as name:
    segment_ids = rt_ids.value_rowids()
    ids = rt_ids.flat_values
    return _embedding_lookup_sparse_impl(
        params,
        segment_ids,
        sp_weights,
        ids,
        combiner,
        ignore_weights,
        name,
    )


def safe_embedding_lookup_sparse(
    embedding_weights: EmbeddingWeights,
    sparse_ids: ragged_tensor.Ragged,
    sparse_weights=None,
    combiner="mean",
    default_id=None,
    name=None,
):
  """Lookup embedding results, accounting for invalid IDs and empty features.

  The partitioned embedding in `embedding_weights` must all be the same shape
  except for the first dimension. The first dimension is allowed to vary as the
  vocabulary size is not necessarily a multiple of `P`.  `embedding_weights`
  may be a `PartitionedVariable` as returned by using
  `tf.compat.v1.get_variable()` with a
  partitioner.

  Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs
  with non-positive weight. For an entry with no features, the embedding vector
  for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

  The ids and weights may be multi-dimensional `SparseTensor`s or
  `RaggedTensor`s with rank of 2. For `SpareTensor`s with left-aligned non-zero
  entries which can be described as `RaggedTensor`s, use of `RaggedTensor`s can
  yield higher performance. Embeddings are always aggregated along the last
  dimension.

  Args:
    embedding_weights: ShadowVariable.
    sp_ids: `RaggedTensor` with rank 2. The rank is not verified for performance
      reasons.
    sparse_weights: `RaggedTensor` of same type and shape as `sparse_ids`,
      containing float weights corresponding to `sparse_ids`, or `None` if all
      weights are assumed to be 1.0.
    combiner: A string specifying how to combine embedding results for each
      entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean" the
      default.
    default_id: The id to use for an entry with no features.
    name: A name for this operation (optional).

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by `sp_ids`, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

    In other words, if

      `shape(combined embedding_weights) = [p0, p1, ..., pm]`

    and

      `shape(sparse_ids) = shape(sparse_weights) = [d0, d1, ..., dn]`

    then

      `shape(output) = [d0, d1, ... dn-1, p1, ..., pm]`.

    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

      ```python
      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id -1, weight 1.0
      [2, 3]: id 1, weight 3.0
      ```

    `default_id` is 0.

    with `combiner`="mean", then the output will be a 3x20 matrix where

      ```python
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = (params[0, :] * 1.0) / 1.0
      output[2, :] = (params[1, :] * 3.0) / 3.0
      ```

  Raises:
    ValueError: if `embedding_weights` is empty.
  """
  ragged_ids = sparse_ids
  ragged_weights = sparse_weights
  embedding_weights.verify_embedding_weights(ragged_ids, ragged_weights)

  with ops.name_scope(name, "embedding_lookup",
                      [ragged_ids, ragged_weights]) as scope:

    if combiner != "sum":
      ragged_ids, ragged_weights = _prune_invalid_weights_ragged(
          ragged_ids, ragged_weights)
    ragged_ids, is_row_empty = _fill_empty_rows(ragged_ids, default_id or 0)
    if ragged_weights is not None:
      ragged_weights, _ = _fill_empty_rows(ragged_weights, 1.0)

    result = embedding_lookup_sparse(
        embedding_weights,
        ragged_ids,
        ragged_weights,
        combiner=combiner,
        name=None if default_id is None else scope,
    )

    if default_id is None:
      # Broadcast is_row_empty to the same shape as embedding_lookup_result,
      # for use in Select.
      is_row_empty = array_ops.tile(
          array_ops.reshape(is_row_empty, [-1, 1]),
          tf.stack([1, array_ops.shape(result)[1]]),
      )

      result = array_ops.where(is_row_empty,
                               array_ops.zeros_like(result),
                               result,
                               name=scope)

    return result


def _prune_invalid_weights_ragged(ids, weights):
  """Prune invalid weights (< 0) from the input ids and weights."""
  if weights is not None:
    is_weights_valid = math_ops.greater(weights.values, 0)
    nrows = ids.nrows()
    # TODO(philipphack): Consider calling ragged_array_ops.boolean_mask once the
    # resulting performance is comparable to array_ops.boolean_mask. Currently,
    # ragged_array_ops.boolean_mask constructs the returned RaggedTensor by
    # calling its from_row_splits method which does not set value_row_ids and
    # requires it to be computed on demand.
    pruned_values = array_ops.boolean_mask_v2(ids.values, is_weights_valid)
    pruned_value_rowids = array_ops.boolean_mask_v2(ids.value_rowids(),
                                                    is_weights_valid)
    ids = ragged_tensor.RaggedTensor.from_value_rowids(pruned_values,
                                                       pruned_value_rowids,
                                                       nrows=nrows,
                                                       validate=False)

    pruned_weights_values = array_ops.boolean_mask_v2(weights.values,
                                                      is_weights_valid)
    weights = ragged_tensor.RaggedTensor.from_value_rowids(
        pruned_weights_values, pruned_value_rowids, nrows=nrows, validate=False)

  return ids, weights
