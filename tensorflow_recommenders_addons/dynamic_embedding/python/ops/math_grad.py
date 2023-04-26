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

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops

from tensorflow_recommenders_addons.utils.resource_loader import prefix_op_name


@ops.RegisterGradient(prefix_op_name("SparseSegmentSum"))
def _TfraSparseSegmentSumGrad(op, grad):
  """Gradient for TFRA>SparseSegmentSum."""
  input_rows = array_ops.shape(op.inputs[0])[0]
  return (math_ops.unsorted_segment_sum(array_ops.gather(grad, op.inputs[2]),
                                        op.inputs[1], input_rows), None, None)


@ops.RegisterGradient(prefix_op_name("SparseSegmentSumWithNumSegments"))
def _TfraSparseSegmentSumWithNumSegmentsGrad(op, grad):
  """Gradient for TFRA>SparseSegmentSumWithNumSegments."""
  input_rows = array_ops.shape(op.inputs[0])[0]
  return (math_ops.unsorted_segment_sum(array_ops.gather(grad, op.inputs[2]),
                                        op.inputs[1],
                                        input_rows), None, None, None)


@ops.RegisterGradient("TfraSparseFillEmptyRows")
def _SparseFillEmptyRowsGrad(op, unused_grad_output_indices, output_grad_values,
                             unused_grad_empty_row_indicator,
                             unused_grad_reverse_index_map):
  """Gradients for TfraSparseFillEmptyRows."""
  reverse_index_map = op.outputs[3]

  d_values, d_default_value = gen_sparse_ops.sparse_fill_empty_rows_grad(
      reverse_index_map=reverse_index_map, grad_values=output_grad_values)

  # d_indices, d_values, d_dense_shape, d_default_value.
  return [None, d_values, None, d_default_value]
