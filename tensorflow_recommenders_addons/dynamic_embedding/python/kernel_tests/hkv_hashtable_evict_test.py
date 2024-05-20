# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = "1"

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
  return tf.add(keys, tf.constant([1], shape=[
      4,
  ], dtype=dtypes.int64))


custom_fun_called = 0


def gen_scores_fn_custom(keys):
  global custom_fun_called
  if custom_fun_called == 0:
    custom_fun_called += 1
    return constant_op.constant(
        np.full(len(keys), 10000).astype(_type_converter(dtypes.int64)),
        dtypes.int64)
  else:
    return constant_op.constant(
        np.full(len(keys), 1).astype(_type_converter(dtypes.int64)),
        dtypes.int64)


class HkvHashtableTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def test_evict_strategy(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    strategy_i = 0
    key_dtype = dtypes.int64
    value_dtype = dtypes.int32
    dim = 8
    for strategy in de.HkvEvictStrategy:
      with self.session(use_gpu=True, config=default_config):
        with self.captureWritesToStream(sys.stderr) as printed:
          table = de.get_variable(
              str(strategy),
              key_dtype=key_dtype,
              value_dtype=value_dtype,
              initializer=0,
              dim=dim,
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
          self.assertTrue(content in printed.contents())
          strategy_i = strategy_i + 1

          keys = constant_op.constant(
              np.array([0, 1, 2, 3]).astype(_type_converter(key_dtype)),
              key_dtype)
          values = constant_op.constant(
              _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim],
                       value_dtype), value_dtype)

          self.evaluate(table.upsert(keys, values))

          output = table.lookup(keys)
          self.assertAllEqual(values, self.evaluate(output))

          del table

  @test_util.run_in_graph_and_eager_modes()
  def test_export_keys_and_scores(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    key_dtype = dtypes.int64
    value_dtype = dtypes.int32
    dim = 8
    for strategy in de.HkvEvictStrategy:
      with self.session(use_gpu=True, config=default_config):
        table = de.get_variable(
            str(strategy),
            key_dtype=key_dtype,
            value_dtype=value_dtype,
            initializer=0,
            dim=dim,
            init_size=1024,
            kv_creator=de.HkvHashTableCreator(
                config=de.HkvHashTableConfig(init_capacity=1024,
                                             max_capacity=1024,
                                             max_hbm_for_values=1024 * 64,
                                             evict_strategy=strategy,
                                             gen_scores_fn=gen_scores_fn)))
        keys = constant_op.constant(
            np.array([0, 1, 2, 3]).astype(_type_converter(key_dtype)),
            key_dtype)
        values = constant_op.constant(
            _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype),
            value_dtype)

        self.evaluate(table.upsert(keys, values))

        exported_keys, exported_scores = self.evaluate(
            table.export_keys_and_scores(1))
        self.assertAllEqual(np.sort(exported_keys), keys)
        if strategy is de.HkvEvictStrategy.CUSTOMIZED:
          self.assertAllEqual(np.sort(exported_scores), gen_scores_fn(keys))
        elif strategy is de.HkvEvictStrategy.EPOCHLFU:
          self.assertAllEqual(exported_scores, np.full((4), 1))
        elif strategy is de.HkvEvictStrategy.LFU:
          self.assertAllEqual(exported_scores, np.ones(4))

        del table

  def test_evict_strategy_lfu(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    key_dtype = dtypes.int64
    value_dtype = dtypes.int32
    dim = 8
    strategy = de.HkvEvictStrategy.LFU
    with self.session(use_gpu=True, config=default_config):
      table = de.get_variable(
          str(strategy),
          key_dtype=key_dtype,
          value_dtype=value_dtype,
          initializer=0,
          dim=dim,
          init_size=1024,
          kv_creator=de.HkvHashTableCreator(
              config=de.HkvHashTableConfig(init_capacity=1024,
                                           max_capacity=1024,
                                           max_hbm_for_values=1024 * 64,
                                           evict_strategy=strategy,
                                           gen_scores_fn=gen_scores_fn)))
      keys = constant_op.constant(
          np.array([0, 1, 2, 3]).astype(_type_converter(key_dtype)), key_dtype)
      values = constant_op.constant(
          _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype),
          value_dtype)

      self.evaluate(table.upsert(keys, values))
      exported_keys, exported_scores = self.evaluate(
          table.export_keys_and_scores(1))
      self.assertAllEqual(exported_scores, np.ones(4))

      self.evaluate(table.upsert(keys, values))
      exported_keys, exported_scores = self.evaluate(
          table.export_keys_and_scores(1))
      self.assertAllEqual(exported_scores, np.full((4), 2))

      keys = constant_op.constant(
          np.array([0, 1, 4, 5]).astype(_type_converter(key_dtype)), key_dtype)

      self.evaluate(table.upsert(keys, values))
      exported_keys, exported_scores = self.evaluate(
          table.export_keys_and_scores(1))
      self.assertAllEqual(np.sort(exported_scores), np.array([1, 1, 2, 2, 3,
                                                              3]))

      keys = constant_op.constant(
          np.arange(4, 1034).astype(_type_converter(key_dtype)), key_dtype)
      values = constant_op.constant(
          _convert([[10] * dim] * len(keys), value_dtype), value_dtype)

      self.evaluate(table.upsert(keys, values))
      exported_keys, exported_scores = self.evaluate(
          table.export_keys_and_scores(1))
      self.assertTrue(len(exported_keys) < 1024)

      higher_frequence_keys = np.arange(0, 6)
      exported_higher_frequence_keys = np.sort(
          exported_keys)[:len(higher_frequence_keys)]
      self.assertTrue(
          np.array_equal(higher_frequence_keys, exported_higher_frequence_keys))

      higher_frequence_scores = np.array([2, 2, 2, 2, 3, 3])
      exported_higher_frequence_scores = np.sort(
          exported_scores)[-len(higher_frequence_keys):]
      self.assertTrue(
          np.array_equal(higher_frequence_scores,
                         exported_higher_frequence_scores))

      del table

  def test_evict_strategy_epoch_lfu(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    key_dtype = dtypes.int64
    value_dtype = dtypes.int32
    dim = 8
    strategy = de.HkvEvictStrategy.EPOCHLFU
    with self.session(use_gpu=True, config=default_config):
      table = de.get_variable(
          str(strategy),
          key_dtype=key_dtype,
          value_dtype=value_dtype,
          initializer=0,
          dim=dim,
          init_size=1024,
          kv_creator=de.HkvHashTableCreator(
              config=de.HkvHashTableConfig(init_capacity=1024,
                                           max_capacity=1024,
                                           max_hbm_for_values=1024 * 64,
                                           evict_strategy=strategy,
                                           step_per_epoch=4,
                                           gen_scores_fn=gen_scores_fn)))

      base_epoch_lfu_scores_list = [1, 1 + (1 << 32), 1 + (2 << 32)]

      for base_epoch_lfu_scores in base_epoch_lfu_scores_list:
        keys = constant_op.constant(
            np.array([0, 1, 2, 3]).astype(_type_converter(key_dtype)),
            key_dtype)
        values = constant_op.constant(
            _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype),
            value_dtype)
        self.evaluate(table.upsert(keys, values))
        exported_keys, exported_scores = self.evaluate(
            table.export_keys_and_scores(1))
        self.assertAllEqual(
            np.greater_equal(
                np.sort(exported_scores)[-4:],
                np.full((4), base_epoch_lfu_scores)), np.full((4), True))

        self.evaluate(table.upsert(keys, values))
        exported_keys, exported_scores = self.evaluate(
            table.export_keys_and_scores(1))
        self.assertAllEqual(
            np.greater_equal(
                np.sort(exported_scores)[-4:],
                np.full((4), base_epoch_lfu_scores + 1)), np.full((4), True))

        keys = constant_op.constant(
            np.array([0, 1, 4, 5]).astype(_type_converter(key_dtype)),
            key_dtype)

        self.evaluate(table.upsert(keys, values))
        exported_keys, exported_scores = self.evaluate(
            table.export_keys_and_scores(1))
        self.assertAllEqual(
            np.greater_equal(
                np.sort(exported_scores)[-6:],
                np.array([
                    base_epoch_lfu_scores, base_epoch_lfu_scores,
                    base_epoch_lfu_scores + 1, base_epoch_lfu_scores + 1,
                    base_epoch_lfu_scores + 2, base_epoch_lfu_scores + 2
                ])), np.full((6), True))

        keys = constant_op.constant(
            np.arange(4, 1024).astype(_type_converter(key_dtype)), key_dtype)
        values = constant_op.constant(
            _convert([[10] * dim] * len(keys), value_dtype), value_dtype)

        self.evaluate(table.upsert(keys, values))
        exported_keys, exported_scores = self.evaluate(
            table.export_keys_and_scores(1))
        self.assertTrue(len(exported_keys) < 1024)

        higher_frequence_keys = np.arange(0, 6)
        exported_higher_frequence_keys = np.sort(
            exported_keys)[:len(higher_frequence_keys)]

        self.assertTrue(
            np.array_equal(higher_frequence_keys,
                           exported_higher_frequence_keys))

        higher_frequence_scores = np.array([
            base_epoch_lfu_scores + 1, base_epoch_lfu_scores + 1,
            base_epoch_lfu_scores + 1, base_epoch_lfu_scores + 1,
            base_epoch_lfu_scores + 2, base_epoch_lfu_scores + 2
        ])
        exported_higher_frequence_scores = np.sort(
            exported_scores)[-len(higher_frequence_keys):]
        self.assertAllEqual(
            np.greater_equal(exported_higher_frequence_scores,
                             higher_frequence_scores),
            np.full((len(higher_frequence_keys)), True))

      del table

  def test_evict_strategy_lru(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    key_dtype = dtypes.int64
    value_dtype = dtypes.int32
    dim = 8
    strategy = de.HkvEvictStrategy.LRU
    with self.session(use_gpu=True, config=default_config):
      table = de.get_variable(
          str(strategy),
          key_dtype=key_dtype,
          value_dtype=value_dtype,
          initializer=0,
          dim=dim,
          init_size=1024,
          kv_creator=de.HkvHashTableCreator(
              config=de.HkvHashTableConfig(init_capacity=1024,
                                           max_capacity=1024,
                                           max_hbm_for_values=1024 * 64,
                                           evict_strategy=strategy,
                                           gen_scores_fn=gen_scores_fn)))
      keys = constant_op.constant(
          np.array([0, 1, 2, 3]).astype(_type_converter(key_dtype)), key_dtype)
      values = constant_op.constant(
          _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype),
          value_dtype)

      self.evaluate(table.upsert(keys, values))
      exported_keys, exported_scores = self.evaluate(
          table.export_keys_and_scores(1))
      self.assertAllEqual(np.isin(keys, exported_keys), np.full((4), True))

      keys = constant_op.constant(
          np.array([2, 3, 6, 7]).astype(_type_converter(key_dtype)), key_dtype)

      self.evaluate(table.upsert(keys, values))
      exported_keys, exported_scores = self.evaluate(
          table.export_keys_and_scores(1))

      l1_scores = []
      l2_scores = []

      for i in range(len(exported_keys)):
        if exported_keys[i] == 0 or exported_keys[i] == 1:
          l1_scores.append(exported_scores[i])
        elif exported_keys[i] == 2 or exported_keys[i] == 3:
          l2_scores.append(exported_scores[i])

      self.assertTrue(l1_scores < l2_scores)
      keys = constant_op.constant(
          np.arange(4, 1044).astype(_type_converter(key_dtype)), key_dtype)
      values = constant_op.constant(
          _convert([[10] * dim] * len(keys), value_dtype), value_dtype)

      self.evaluate(table.upsert(keys, values))
      exported_keys, exported_scores = self.evaluate(
          table.export_keys_and_scores(1))

      keys = constant_op.constant(
          np.arange(1024, 1400).astype(_type_converter(key_dtype)), key_dtype)
      values = constant_op.constant(
          _convert([[10] * dim] * len(keys), value_dtype), value_dtype)
      self.evaluate(table.upsert(keys, values))
      exported_keys, exported_scores = self.evaluate(
          table.export_keys_and_scores(1))
      self.assertTrue(len(exported_keys) <= 1024)

      evicted_keys = np.arange(0, 4)
      self.assertAllEqual(np.isin(evicted_keys, exported_keys),
                          np.array([False, False, False, False]))

      del table

  def test_evict_strategy_epoch_lru(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    key_dtype = dtypes.int64
    value_dtype = dtypes.int32
    dim = 8
    strategy = de.HkvEvictStrategy.EPOCHLRU
    with self.session(use_gpu=True, config=default_config):
      table = de.get_variable(
          str(strategy),
          key_dtype=key_dtype,
          value_dtype=value_dtype,
          initializer=0,
          dim=dim,
          init_size=1024,
          kv_creator=de.HkvHashTableCreator(
              config=de.HkvHashTableConfig(init_capacity=1024,
                                           max_capacity=1024,
                                           max_hbm_for_values=1024 * 64,
                                           evict_strategy=strategy,
                                           step_per_epoch=1,
                                           gen_scores_fn=gen_scores_fn,
                                           reserved_key_start_bit=1)))

      base_epoch_lfu_scores_list = [1, 1 + (1 << 32), 1 + (2 << 32)]

      for epoch in range(2):
        keys = constant_op.constant(
            np.arange(0, 1024).astype(_type_converter(key_dtype)), key_dtype)
        values = constant_op.constant(
            _convert([[10] * dim] * len(keys), value_dtype), value_dtype)

        self.evaluate(table.upsert(keys, values))
        exported_keys, exported_scores = self.evaluate(
            table.export_keys_and_scores(1))

        self.assertAllEqual(
            np.greater(np.full(len(exported_scores), (epoch << 32)),
                       exported_scores), np.full(len(exported_scores), False))
        self.assertAllEqual(
            np.greater(
                np.full(len(exported_scores), (epoch << 32) + 0xffffffff),
                exported_scores), np.full(len(exported_scores), True))

      del table

  def test_evict_strategy_custom(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    key_dtype = dtypes.int64
    value_dtype = dtypes.int32
    dim = 8
    strategy = de.HkvEvictStrategy.CUSTOMIZED
    with self.session(use_gpu=True, config=default_config):
      table = de.get_variable(
          str(strategy),
          key_dtype=key_dtype,
          value_dtype=value_dtype,
          initializer=0,
          dim=dim,
          init_size=1024,
          kv_creator=de.HkvHashTableCreator(
              config=de.HkvHashTableConfig(init_capacity=1024,
                                           max_capacity=1024,
                                           max_hbm_for_values=1024 * 64,
                                           evict_strategy=strategy,
                                           gen_scores_fn=gen_scores_fn_custom)))

      base_epoch_lfu_scores_list = [1, 1 + (1 << 32), 1 + (2 << 32)]

      keys = constant_op.constant(
          np.arange(2048, 4096).astype(_type_converter(key_dtype)), key_dtype)
      values = constant_op.constant(
          _convert([[10] * dim] * len(keys), value_dtype), value_dtype)

      self.evaluate(table.upsert(keys, values))

      keys = constant_op.constant(
          np.arange(0, 1024).astype(_type_converter(key_dtype)), key_dtype)
      values = constant_op.constant(
          _convert([[10] * dim] * len(keys), value_dtype), value_dtype)
      self.evaluate(table.upsert(keys, values))

      exported_keys, exported_scores = self.evaluate(
          table.export_keys_and_scores(1))
      self.assertAllEqual(
          np.equal(np.full(len(exported_scores), 10000), exported_scores),
          np.full(len(exported_scores), True))
      self.assertAllEqual(
          np.greater(np.full(len(exported_keys), 1024), exported_keys),
          np.full(len(exported_keys), False))

      del table


if __name__ == "__main__":
  test.main()
