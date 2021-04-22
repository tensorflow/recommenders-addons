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

from tensorflow_recommenders_addons.dynamic_embedding.python.ops import math_ops as de_math

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

default_config = config_pb2.ConfigProto(
    allow_soft_placement=False,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))

use_gpu = test_util.is_gpu_available()


class SparseSemgentReductionOpsTest(object):

  def forward_compute(self, data, indices, segment_ids, num_segments=None):
    result = de_math.sparse_segment_sum(data,
                                        indices,
                                        segment_ids,
                                        num_segments=num_segments)
    expected = math_ops.sparse_segment_sum(data,
                                           indices,
                                           segment_ids,
                                           num_segments=num_segments)
    return result, expected


class SparseSegmentSumGpuTest(test.TestCase, SparseSemgentReductionOpsTest):

  @test_util.run_in_graph_and_eager_modes
  def test_not_equal_indices_and_seg_ids_num(self):
    with self.session(use_gpu=use_gpu, config=default_config):
      data = constant_op.constant(list(range(20)), dtype=dtypes.float32)
      data = array_ops.reshape(data, (10, 2))
      indices = constant_op.constant(list(range(6)), dtype=dtypes.int32)
      segment_ids = constant_op.constant(list(range(7)), dtype=dtypes.int32)
      with self.assertRaises((ValueError, errors.InvalidArgumentError)):
        target = de_math.sparse_segment_sum(data, indices, segment_ids)
        self.evaluate(target)

  @test_util.run_in_graph_and_eager_modes
  def test_value(self):
    with self.session(use_gpu=use_gpu, config=default_config):
      data = constant_op.constant(
          [[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=dtypes.float32)
      indices = constant_op.constant([0, 1], dtype=dtypes.int32)
      segment_ids = constant_op.constant([0, 5], dtype=dtypes.int32)
      result, expected = self.forward_compute(data, indices, segment_ids)
      self.assertAllEqual(self.evaluate(result), self.evaluate(expected))


class SparseSegmentSumWithNumSegmentsGpuTest(test.TestCase,
                                             SparseSemgentReductionOpsTest):

  @test_util.run_in_graph_and_eager_modes
  def test_not_equal_indices_and_seg_ids_num(self):
    with self.session(use_gpu=use_gpu, config=default_config):
      data = constant_op.constant(list(range(20)), dtype=dtypes.float32)
      data = array_ops.reshape(data, (10, 2))
      indices = constant_op.constant(list(range(6)), dtype=dtypes.int32)
      segment_ids = constant_op.constant(list(range(7)), dtype=dtypes.int32)
      num_segments = 100
      with self.assertRaises((ValueError, errors.InvalidArgumentError)):
        target = de_math.sparse_segment_sum(data,
                                            indices,
                                            segment_ids,
                                            num_segments=num_segments)
        self.evaluate(target)

  @test_util.run_in_graph_and_eager_modes
  def test_value(self):
    with self.session(use_gpu=use_gpu, config=default_config):
      data = constant_op.constant(
          [[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=dtypes.float32)
      indices = constant_op.constant([0, 1], dtype=dtypes.int32)
      segment_ids = constant_op.constant([0, 5], dtype=dtypes.int32)
      num_segments = 100
      result, expected = self.forward_compute(data,
                                              indices,
                                              segment_ids,
                                              num_segments=num_segments)

      self.assertAllEqual(self.evaluate(result), self.evaluate(expected))


if __name__ == "__main__":
  test.main()
