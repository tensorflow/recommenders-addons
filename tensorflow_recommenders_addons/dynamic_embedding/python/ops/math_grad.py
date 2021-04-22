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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops


@ops.RegisterGradient("TFRA>SparseSegmentSum")
def _TfraSparseSegmentSumGrad(op, grad):
  """Gradient for TFRA>SparseSegmentSum."""
  input_rows = array_ops.shape(op.inputs[0])[0]
  return (math_ops.unsorted_segment_sum(array_ops.gather(grad, op.inputs[2]),
                                        op.inputs[1], input_rows), None, None)


@ops.RegisterGradient("TFRA>SparseSegmentSumWithNumSegments")
def _TfraSparseSegmentSumWithNumSegmentsGrad(op, grad):
  """Gradient for TFRA>SparseSegmentSumWithNumSegments."""
  input_rows = array_ops.shape(op.inputs[0])[0]
  return (math_ops.unsorted_segment_sum(array_ops.gather(grad, op.inputs[2]),
                                        op.inputs[1],
                                        input_rows), None, None, None)
