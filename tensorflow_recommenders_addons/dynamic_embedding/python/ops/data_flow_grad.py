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
"""gradients for math operations."""
# pylint: disable=g-bad-name

from tensorflow.python.framework import dtypes
try:  # tf version >= 2.13.0
  from tensorflow.python.framework.indexed_slices import IndexedSlices
except:
  from tensorflow.python.framework.ops import IndexedSlices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from tensorflow_recommenders_addons.dynamic_embedding.python.ops import data_flow_ops as de_data_flow
from tensorflow_recommenders_addons.utils.resource_loader import prefix_op_name


@ops.RegisterGradient(prefix_op_name("DynamicPartition"))
def _TfraDynamicPartitionGrads(op, *grads):
  """Gradients for TfraDynamicPartition."""
  data = op.inputs[0]
  indices = op.inputs[1]
  num_partitions = op.get_attr("num_partitions")

  prefix_shape = array_ops.shape(indices)
  original_indices = array_ops.reshape(
      math_ops.range(math_ops.reduce_prod(prefix_shape)), prefix_shape)
  partitioned_indices = de_data_flow.dynamic_partition(original_indices,
                                                       indices, num_partitions)
  reconstructed = de_data_flow.dynamic_stitch(partitioned_indices, grads)
  reconstructed = array_ops.reshape(reconstructed, array_ops.shape(data))
  return [reconstructed, None]


@ops.RegisterGradient(prefix_op_name("DynamicStitch"))
@ops.RegisterGradient(prefix_op_name("DynamicStitchFast"))
@ops.RegisterGradient(prefix_op_name("ParallelDynamicStitch"))
def _TfraDynamicStitchGrads(op, grad):
  """Gradients for TfraDynamicStitch/TfraDynamicStitchFast and TfraParallelDynamicStitch."""

  num_values = len(op.inputs) // 2
  indices_grad = [None] * num_values

  def AsInt32(x):
    return (x if op.inputs[0].dtype == dtypes.int32 else math_ops.cast(
        x, dtypes.int32))

  inputs = [AsInt32(op.inputs[i]) for i in range(num_values)]
  if isinstance(grad, IndexedSlices):
    output_shape = array_ops.shape(op.outputs[0])
    output_rows = output_shape[0]
    grad = math_ops.unsorted_segment_sum(grad.values, grad.indices, output_rows)
  values_grad = [array_ops.gather(grad, inp) for inp in inputs]
  return indices_grad + values_grad
