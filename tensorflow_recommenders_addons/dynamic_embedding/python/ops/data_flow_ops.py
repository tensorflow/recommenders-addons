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
"""data flow operations."""
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
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging

from tensorflow_recommenders_addons.dynamic_embedding.python.ops import data_flow_grad
from tensorflow_recommenders_addons.utils.resource_loader import LazySO

tfra_data_flow_ops = LazySO("dynamic_embedding/core/_data_flow_ops.so").ops
# TODO(MoFHeka): dynamic_partition and dynamic_stitch in TensorFlow
#                missing compile implementation of the bool data type.
#                Remove TFRA DE same-name op when TensorFlow supports those data type.


def dynamic_partition(data, partitions, num_partitions, name=None):
  if not hasattr(tfra_data_flow_ops, 'tfra_dynamic_partition'):
    tf_logging.warn(
        '`tfra.dynamic_embedding.data_flow.dynamic_partition` is not'
        ' found. Use tf.dynamic_partition instead.')
    with ops.colocate_with(None, ignore_existing=True):
      return tf.dynamic_partition(data, partitions, num_partitions, name=name)
  else:
    return tfra_data_flow_ops.tfra_dynamic_partition(data,
                                                     partitions,
                                                     num_partitions,
                                                     name=name)


def dynamic_stitch(indices, data, use_fast=True, name=None):
  if not hasattr(tfra_data_flow_ops, 'tfra_dynamic_stitch'):
    tf_logging.warn('`tfra.dynamic_embedding.data_flow.dynamic_stitch` is not'
                    ' found. Use tf.dynamic_stitch instead.')
    with ops.colocate_with(None, ignore_existing=True):
      return tf.dynamic_stitch(indices, data, name=name)
  else:
    if use_fast is True:
      return tfra_data_flow_ops.tfra_dynamic_stitch_fast(indices,
                                                         data,
                                                         name=name)
    else:
      return tfra_data_flow_ops.tfra_dynamic_stitch(indices, data, name=name)
