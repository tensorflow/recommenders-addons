# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""math operations."""
# pylint: disable=g-bad-name

import functools
import numpy as np
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.saver import BaseSaverBuilder

from tensorflow_recommenders_addons.utils.resource_loader import LazySO
from tensorflow_recommenders_addons.dynamic_embedding.python.ops import math_grad

tfra_math_ops = LazySO("dynamic_embedding/core/_math_ops.so").ops


def _convert_to_sparse_tensor(sp_input):
  """Convert `sp_input` to `SparseTensor` and return it.

  Args:
    sp_input: `SparseTensor` or `SparseTensorValue`.

  Returns:
    `sp_input` converted to `SparseTensor`.

  Raises:
    ValueError: if `sp_input` is neither `SparseTensor` nor `SparseTensorValue`.
  """
  if isinstance(sp_input, sparse_tensor.SparseTensorValue):
    return sparse_tensor.SparseTensor.from_value(sp_input)
  if not isinstance(sp_input, sparse_tensor.SparseTensor):
    raise TypeError("Input must be a SparseTensor.")
  return sp_input


def sparse_segment_sum(data,
                       indices,
                       segment_ids,
                       name=None,
                       num_segments=None):
  """
  Computes the sum along sparse segments of a tensor. It do same things
  as `tf.sparse.segment_sum`. Here we provide GPU impl.

  Go [tf api](https://www.tensorflow.org/api_docs/python/tf/sparse/segment_sum)
  for more details.

  Args:
    data: A `Tensor` with data that will be assembled in the output.
    indices: A 1-D `Tensor` with indices into `data`. Has same rank as
      `segment_ids`.
    segment_ids: A 1-D `Tensor` with indices into the output `Tensor`. Values
      should be sorted and can be repeated.
    name: A name for the operation (optional).
    num_segments: An optional int32 scalar. Indicates the size of the output
      `Tensor`.

  Returns:
    A `tensor` of the shape as data, except for dimension 0 which
    has size `k`, the number of segments specified via `num_segments` or
    inferred for the last element in `segments_ids`.
  """
  gpu_devices = config.list_physical_devices('GPU')
  if gpu_devices:
    if context.executing_eagerly():
      try:
        return _sparse_segment_sum_gpu(data,
                                       indices,
                                       segment_ids,
                                       name=name,
                                       num_segments=num_segments)
      except errors.NotFoundError:
        tf_logging.warn(
            '`tfra.dynamic_embedding.math.sparse_segment_sum` is not'
            ' found. Use tf.sparse.segment_sum instead.')
        return tf.sparse.segment_sum(data,
                                     indices,
                                     segment_ids,
                                     name=name,
                                     num_segments=num_segments)

    else:
      predef = _sparse_segment_sum_gpu(data,
                                       indices,
                                       segment_ids,
                                       name=name,
                                       num_segments=num_segments)

      use_origin = False
      if predef.device == '':
        tf_logging.warn(
            'SparseSegmentSum({}) has not been assigned device, '
            'while GPU are available: {}, so use GPU by default.'.format(
                predef.name, gpu_devices))
      else:
        device_type = predef.device.split(':')[-2][-3:].lower()
        if 'gpu' in device_type:
          use_origin = True

      if use_origin:
        return tf.sparse.segment_sum(data,
                                     indices,
                                     segment_ids,
                                     name=name,
                                     num_segments=num_segments)
      return predef

  else:
    return tf.sparse.segment_sum(data,
                                 indices,
                                 segment_ids,
                                 name=name,
                                 num_segments=num_segments)


def _sparse_segment_sum_gpu(data,
                            indices,
                            segment_ids,
                            name=None,
                            num_segments=None):
  if not hasattr(tfra_math_ops, 'tfra_sparse_segment_sum'):
    tf_logging.warn('`tfra.dynamic_embedding.math.sparse_segment_sum` is not'
                    ' found. Use tf.sparse.segment_sum instead.')
    return tf.sparse.segment_sum(data,
                                 indices,
                                 segment_ids,
                                 name=name,
                                 num_segments=num_segments)

  if num_segments is not None:
    return tfra_math_ops.tfra_sparse_segment_sum_with_num_segments(
        data=data,
        indices=indices,
        segment_ids=segment_ids,
        name=name,
        num_segments=num_segments)
  else:
    return tfra_math_ops.tfra_sparse_segment_sum(data=data,
                                                 indices=indices,
                                                 segment_ids=segment_ids,
                                                 name=name)


def sparse_fill_empty_rows(sp_input, default_value, name=None):
  """Fills empty rows in the input 2-D `SparseTensor` with a default value. 

  It does same things as `tf.sparse.fill_empty_rows`. Here we provide GPU implement.

  Go [TF API](https://www.tensorflow.org/api_docs/python/tf/sparse/fill_empty_rows)
  for more details.

  Args:
    sp_input: A `SparseTensor` with shape `[N, M]`.
    default_value: The value to fill for empty rows, with the same type as
      `sp_input`.
    name: A name prefix for the returned tensors (optional).

  Returns:
    sp_ordered_output: A `SparseTensor` with shape `[N, M]`, and with all empty
      rows filled in with `default_value`.
    empty_row_indicator: A bool vector of length `N` indicating whether each
      input row was empty.
  """
  gpu_devices = config.list_physical_devices('GPU')
  return tf.sparse.fill_empty_rows(sp_input, default_value, name=name)


def sparse_reshape(sp_input, shape, name=None):
  """Reshapes a `SparseTensor` to represent values in a new dense shape.

  It does same things as `tf.sparse.reshape`. Here we provide GPU implement.

  Go [TF API](https://www.tensorflow.org/api_docs/python/tf/sparse/reshape)
  for more details.

  Args:
    sp_input: The input `SparseTensor`.
    shape: A 1-D (vector) int64 `Tensor` specifying the new dense shape of the
      represented `SparseTensor`.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A `SparseTensor` with the same non-empty values but with indices calculated
    by the new dense shape.
  """
  gpu_devices = config.list_physical_devices('GPU')
  if gpu_devices and hasattr(sp_input, 'device'):
    # Deprecated(MoFHeka): tfra_sparse_reshape op cause core dump when import horovod with GPU.
    if context.executing_eagerly():
      with ops.colocate_with(sp_input, ignore_existing=True):
        return tf.sparse.reshape(sp_input, shape, name=name)
    else:
      with ops.colocate_with(sp_input, ignore_existing=True):
        predef = tf.sparse.reshape(sp_input, shape, name=name)

      if predef.values.device == '':
        tf_logging.warn(
            'SparseReshape({}) has not been assigned device, '
            'while GPU are available: {}, so use GPU by default.'.format(
                predef.values.name, gpu_devices))
        with ops.device('gpu'):
          predef = tf.sparse.reshape(sp_input, shape, name=name)

      return predef
  else:
    return tf.sparse.reshape(sp_input, shape, name=name)
