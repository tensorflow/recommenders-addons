# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""unit tests of hkv hashtable ops
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import itertools
import numpy as np

from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.utils.check_platform import is_windows, is_macos, is_arm64, is_linux, is_raspi_arm

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
from tensorflow.python.client import session

import tensorflow as tf
try:
  import tensorflow_io
except:
  print()


def _type_converter(tf_type):
  mapper = {
      dtypes.int32: np.int32,
      dtypes.int64: np.int64,
      dtypes.float32: float,
      dtypes.float64: np.float64,
      dtypes.string: str,
      dtypes.half: np.float16,
      dtypes.int8: np.int8,
      dtypes.bool: bool,
  }
  return mapper[tf_type]


def _convert(v, t):
  return np.array(v).astype(_type_converter(t))


default_config = config_pb2.ConfigProto(
    allow_soft_placement=False,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))


def _get_devices():
  return ["/gpu:0" if test_util.is_gpu_available() else "/cpu:0"]


is_gpu_available = test_util.is_gpu_available()


def convert(v, t):
  return np.array(v).astype(_type_converter(t))


def gen_scores_fn(keys):
  return tf.constant([1, 2, 3, 4], dtypes.int64)


class HkvHashtableTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def test_evict_strategy(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    strategy_i = 0
    for strategy in de.HkvEvictStrategy:
      with self.session(use_gpu=True, config=default_config):
        with self.captureWritesToStream(sys.stderr) as printed:
          table = de.get_variable(
              str(strategy),
              key_dtype=dtypes.int64,
              value_dtype=dtypes.int32,
              initializer=0,
              dim=8,
              init_size=1024,
              kv_creator=de.HkvHashTableCreator(
                  config=de.HkvHashTableConfig(init_capacity=1024,
                                               max_capacity=1024,
                                               max_hbm_for_values=1024 * 4 * 8 *
                                               2,
                                               evict_strategy=strategy,
                                               gen_scores_fn=gen_scores_fn)))
          self.evaluate(table.size())

          content = "Use Evict Strategy:" + str(strategy_i)
          # self.assertTrue(content in printed.contents())
          strategy_i = strategy_i + 1

          key_dtype = dtypes.int64
          value_dtype = dtypes.int32
          dim = 8

          keys = constant_op.constant(
              np.array([0, 1, 2, 3]).astype(_type_converter(key_dtype)),
              key_dtype)
          values = constant_op.constant(
              _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim],
                       value_dtype), value_dtype)

          self.evaluate(table.upsert(keys, values))

          output = table.lookup(keys)
          self.assertAllEqual(values, self.evaluate(output))

          # exported_keys, exported_scores = self.evaluate(table.export_keys_and_scores())
          # print(exported_keys)
          # print(exported_scores)

          del table


if __name__ == "__main__":
  test.main()
