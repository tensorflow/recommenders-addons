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
"""unit tests of cuckoo hashtable ops
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow_recommenders_addons.dynamic_embedding.python.ops import math_ops as de_math

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test

default_config = config_pb2.ConfigProto(
    allow_soft_placement=False,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))

use_gpu = test_util.is_gpu_available()


class SparseSemgentReductionGradOpsTestBase(object):

  def backward_compute(self, data, indices, segment_ids, num_segments=None):
    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(data)
      result = de_math.sparse_segment_sum(data,
                                          indices,
                                          segment_ids,
                                          num_segments=num_segments)
      expected = math_ops.sparse_segment_sum(data,
                                             indices,
                                             segment_ids,
                                             num_segments=num_segments)
    result = tape.gradient(result, data)
    expected = tape.gradient(expected, data)
    return result, expected


class SparseSegmentSumGpuGradTest(test.TestCase,
                                  SparseSemgentReductionGradOpsTestBase):

  @test_util.run_in_graph_and_eager_modes
  def test_value(self):
    with self.session(use_gpu=use_gpu, config=default_config):
      data = constant_op.constant(
          [[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=dtypes.float32)
      indices = constant_op.constant([0, 1], dtype=dtypes.int32)
      segment_ids = constant_op.constant([0, 5], dtype=dtypes.int32)
      result, expected = self.backward_compute(data, indices, segment_ids)

      self.assertAllEqual(self.evaluate(result), self.evaluate(expected))


class SparseSegmentSumGpuGradTest(test.TestCase,
                                  SparseSemgentReductionGradOpsTestBase):

  @test_util.run_in_graph_and_eager_modes
  def test_value(self):
    with self.session(use_gpu=use_gpu, config=default_config):
      data = constant_op.constant(
          [[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=dtypes.float32)
      indices = constant_op.constant([0, 1], dtype=dtypes.int32)
      segment_ids = constant_op.constant([0, 5], dtype=dtypes.int32)
      num_segments = 100
      result, expected = self.backward_compute(data,
                                               indices,
                                               segment_ids,
                                               num_segments=num_segments)

      self.assertAllEqual(self.evaluate(result), self.evaluate(expected))


class SparseFillEmptyRowsGpuGradTest(test.TestCase):

  def _SparseTensor_5x6(self):
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    return sparse_tensor.SparseTensor(constant_op.constant(ind, dtypes.int64),
                                      constant_op.constant(val, dtypes.float32),
                                      constant_op.constant(shape, dtypes.int64))

  def backward_compute(self, sp_input, default_value):
    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(sp_input.values)
      result_output, result_indicator = de_math.sparse_fill_empty_rows(
          sp_input, default_value)
      expected_output, expected_indicator = sparse_ops.sparse_fill_empty_rows(
          sp_input, default_value)
    result = tape.gradient(result_output.values, sp_input.values)
    expected = tape.gradient(expected_output.values, sp_input.values)
    return result, expected

  @test_util.run_in_graph_and_eager_modes
  def test_value(self):
    with self.session(use_gpu=use_gpu, config=default_config):
      result, expected = self.backward_compute(self._SparseTensor_5x6(), -1)
      self.assertAllEqual(self.evaluate(result), self.evaluate(expected))


if __name__ == "__main__":
  test.main()
