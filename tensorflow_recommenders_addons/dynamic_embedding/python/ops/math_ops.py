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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.saver import BaseSaverBuilder

from tensorflow_recommenders_addons.utils.resource_loader import LazySO
from tensorflow_recommenders_addons.dynamic_embedding.python.ops import math_grad

segment_reduction_ops = LazySO(
    "dynamic_embedding/core/_segment_reduction_ops.so").ops


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
        tf_logging.warn('`tfra.dynamic_embedding.sparse_segment_sum` is not'
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
      use_origin = True
      if predef.device == '':
        tf_logging.warn(
            'Haven\'t specify devices while GPU devices are'
            'available: {}, use CPU by default.'.format(gpu_devices))
      else:
        device_type = predef.device.split(':')[-2][-3:].lower()
        if device_type == 'gpu':
          use_origin = False

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
  if not hasattr(segment_reduction_ops, 'tfra_sparse_segment_sum'):
    tf_logging.warn('`tfra.dynamic_embedding.sparse_segment_sum` is not'
                    ' found. Use tf.sparse.segment_sum instead.')
    return tf.sparse.segment_sum(data,
                                 indices,
                                 segment_ids,
                                 name=name,
                                 num_segments=num_segments)

  if num_segments is not None:
    return segment_reduction_ops.tfra_sparse_segment_sum_with_num_segments(
        data=data,
        indices=indices,
        segment_ids=segment_ids,
        name=name,
        num_segments=num_segments)
  else:
    return segment_reduction_ops.tfra_sparse_segment_sum(
        data=data, indices=indices, segment_ids=segment_ids, name=name)
