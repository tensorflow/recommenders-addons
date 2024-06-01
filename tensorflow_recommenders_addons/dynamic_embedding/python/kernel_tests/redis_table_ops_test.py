# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""unit tests of variable
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import itertools
import json
import math
import numpy as np
import os
import six
import tempfile

from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.utils.check_platform import is_windows, is_macos, is_arm64, is_linux, is_raspi_arm

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.keras import layers
from tensorflow.python.keras import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat


# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
def _type_converter(tf_type):
  mapper = {
      dtypes.int32: np.int32,
      dtypes.int64: np.int64,
      dtypes.float32: np.float32 if hasattr(np, "float32") else np.float,
      dtypes.float64: np.float64,
      dtypes.string: np.str if hasattr(np, "str") else str,
      dtypes.half: np.float16,
      dtypes.int8: np.int8,
      dtypes.bool: np.bool if hasattr(np, "bool") else bool,
  }
  return mapper[tf_type]


def _get_devices():
  return ["/gpu:0" if test_util.is_gpu_available() else "/cpu:0"]


def _check_device(op, expexted_device="gpu"):
  return expexted_device.upper() in op.device


def embedding_result(params, id_vals, weight_vals=None):
  if weight_vals is None:
    weight_vals = np.copy(id_vals)
    weight_vals.fill(1)
  values = []
  weights = []
  weights_squared = []
  for pms, ids, wts in zip(params, id_vals, weight_vals):
    value_aggregation = None
    weight_aggregation = None
    squared_weight_aggregation = None
    if isinstance(ids, compat.integral_types):
      pms = [pms]
      ids = [ids]
      wts = [wts]
    for val, i, weight_value in zip(pms, ids, wts):
      if value_aggregation is None:
        assert weight_aggregation is None
        assert squared_weight_aggregation is None
        value_aggregation = val * weight_value
        weight_aggregation = weight_value
        squared_weight_aggregation = weight_value * weight_value
      else:
        assert weight_aggregation is not None
        assert squared_weight_aggregation is not None
        value_aggregation += val * weight_value
        weight_aggregation += weight_value
        squared_weight_aggregation += weight_value * weight_value
    values.append(value_aggregation)
    weights.append(weight_aggregation)
    weights_squared.append(squared_weight_aggregation)
  values = np.array(values).astype(np.float32)
  weights = np.array(weights).astype(np.float32)
  weights_squared = np.array(weights_squared).astype(np.float32)
  return values, weights, weights_squared


def data_fn(shape, maxval):
  return random_ops.random_uniform(shape, maxval=maxval, dtype=dtypes.int64)


def model_fn(sparse_vars, embed_dim, feature_inputs):
  embedding_weights = []
  embedding_trainables = []
  for sp in sparse_vars:
    for inp_tensor in feature_inputs:
      embed_w, trainable = de.embedding_lookup(sp,
                                               inp_tensor,
                                               return_trainable=True,
                                               name='nt9527')
      embedding_weights.append(embed_w)
      embedding_trainables.append(trainable)

  def layer_fn(entry, dimension, activation=False):
    entry = array_ops.reshape(entry, (-1, dimension, embed_dim))
    dnn_fn = layers.Dense(dimension, use_bias=False)
    batch_normal_fn = layers.BatchNormalization()
    dnn_result = dnn_fn(entry)
    if activation:
      return batch_normal_fn(nn.selu(dnn_result))
    return dnn_result

  def dnn_fn(entry, dimension, activation=False):
    hidden = layer_fn(entry, dimension, activation)
    output = layer_fn(hidden, 1)
    logits = math_ops.reduce_mean(output)
    return logits

  logits_sum = sum(dnn_fn(w, 16, activation=True) for w in embedding_weights)
  labels = 0.0
  err_prob = nn.sigmoid_cross_entropy_with_logits(logits=logits_sum,
                                                  labels=labels)
  loss = math_ops.reduce_mean(err_prob)
  return labels, embedding_trainables, loss


def ids_and_weights_2d(embed_dim=4):
  # Each row demonstrates a test case:
  #   Row 0: multiple valid ids, 1 invalid id, weighted mean
  #   Row 1: all ids are invalid (leaving no valid ids after pruning)
  #   Row 2: no ids to begin with
  #   Row 3: single id
  #   Row 4: all ids have <=0 weight
  indices = [[0, 0], [0, 1], [0, 2], [1, 0], [3, 0], [4, 0], [4, 1]]
  ids = [0, 1, -1, -1, 2, 0, 1]
  weights = [1.0, 2.0, 1.0, 1.0, 3.0, 0.0, -0.5]
  shape = [5, embed_dim]

  sparse_ids = sparse_tensor.SparseTensor(
      constant_op.constant(indices, dtypes.int64),
      constant_op.constant(ids, dtypes.int64),
      constant_op.constant(shape, dtypes.int64),
  )

  sparse_weights = sparse_tensor.SparseTensor(
      constant_op.constant(indices, dtypes.int64),
      constant_op.constant(weights, dtypes.float32),
      constant_op.constant(shape, dtypes.int64),
  )

  return sparse_ids, sparse_weights


def ids_and_weights_3d(embed_dim=4):
  # Each (2-D) index demonstrates a test case:
  #   Index 0, 0: multiple valid ids, 1 invalid id, weighted mean
  #   Index 0, 1: all ids are invalid (leaving no valid ids after pruning)
  #   Index 0, 2: no ids to begin with
  #   Index 1, 0: single id
  #   Index 1, 1: all ids have <=0 weight
  #   Index 1, 2: no ids to begin with
  indices = [
      [0, 0, 0],
      [0, 0, 1],
      [0, 0, 2],
      [0, 1, 0],
      [1, 0, 0],
      [1, 1, 0],
      [1, 1, 1],
  ]
  ids = [0, 1, -1, -1, 2, 0, 1]
  weights = [1.0, 2.0, 1.0, 1.0, 3.0, 0.0, -0.5]
  shape = [2, 3, embed_dim]

  sparse_ids = sparse_tensor.SparseTensor(
      constant_op.constant(indices, dtypes.int64),
      constant_op.constant(ids, dtypes.int64),
      constant_op.constant(shape, dtypes.int64),
  )

  sparse_weights = sparse_tensor.SparseTensor(
      constant_op.constant(indices, dtypes.int64),
      constant_op.constant(weights, dtypes.float32),
      constant_op.constant(shape, dtypes.int64),
  )

  return sparse_ids, sparse_weights


def _random_weights(
    key_dtype=dtypes.int64,
    value_dtype=dtypes.float32,
    vocab_size=4,
    embed_dim=4,
    num_shards=1,
):
  assert vocab_size > 0
  assert embed_dim > 0
  assert num_shards > 0
  assert num_shards <= vocab_size

  initializer = init_ops.truncated_normal_initializer(mean=0.0,
                                                      stddev=1.0 /
                                                      math.sqrt(vocab_size),
                                                      dtype=dtypes.float32)
  embedding_weights = de.get_variable(
      key_dtype=key_dtype,
      value_dtype=value_dtype,
      devices=_get_devices() * num_shards,
      name="embedding_weights",
      initializer=initializer,
      dim=embed_dim,
  )
  return embedding_weights


def _test_dir(temp_dir, test_name):
  """Create an empty dir to use for tests.

    Args:
      temp_dir: Tmp directory path.
      test_name: Name of the test.

    Returns:
      Absolute path to the test directory.
    """
  test_dir = os.path.join(temp_dir, test_name)
  if os.path.isdir(test_dir):
    for f in glob.glob("%s/*" % test_dir):
      os.remove(f)
  else:
    os.makedirs(test_dir)
  return test_dir


def _create_dynamic_shape_tensor(
    max_len=100,
    min_len=2,
    min_val=0x0000F00000000001,
    max_val=0x0000F00000000020,
    dtype=np.int64,
):

  def _func():
    length = np.random.randint(min_len, max_len)
    tensor = np.random.randint(min_val, max_val, max_len, dtype=dtype)
    tensor = np.array(tensor[0:length], dtype=dtype)
    return tensor

  return _func


default_config = config_pb2.ConfigProto(
    allow_soft_placement=False,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))

redis_config_dir = os.path.join(
    tempfile.mkdtemp(dir=os.environ.get('TEST_TMPDIR')), "save_restore")
redis_config_path = os.path.join(tempfile.mkdtemp(prefix=redis_config_dir),
                                 "hash")
os.makedirs(redis_config_path)
redis_config_path = os.path.join(redis_config_path, "redis_config.json")
redis_config_params = {
    "redis_host_ip": ["127.0.0.1"],
    "redis_host_port": [6379],
    "table_store_mode": 0
}
with open(redis_config_path, 'w', encoding='utf-8') as f:
  f.write(json.dumps(redis_config_params, indent=2, ensure_ascii=True))
redis_config = de.RedisTableConfig(redis_config_abs_dir=redis_config_path)

redis_config_dir = os.path.join(
    tempfile.mkdtemp(dir=os.environ.get('TEST_TMPDIR')), "save_restore")
redis_config_path = os.path.join(tempfile.mkdtemp(prefix=redis_config_dir),
                                 "hash")
os.makedirs(redis_config_path)
redis_config_path = os.path.join(redis_config_path, "redis_config.json")
redis_config_params = {
    "redis_host_ip": ["127.0.0.1"],
    "redis_host_port": [6379],
    "table_store_mode": 0
}
with open(redis_config_path, 'w', encoding='utf-8') as f:
  f.write(json.dumps(redis_config_params, indent=2, ensure_ascii=True))
redis_config = de.RedisTableConfig(redis_config_abs_dir=redis_config_path)


def _redis_health_check(redis_host_ip="127.0.0.1", redis_host_port=6379):
  ping_return = os.popen('redis-cli -h ' + redis_host_ip + ' -p ' +
                         str(redis_host_port) + ' ping').read()
  return ping_return == 'PONG\n'


@test_util.run_all_in_graph_and_eager_modes
class RedisVariableTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def test_basic(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    with self.session(use_gpu=False, config=default_config):
      table = de.get_variable(
          "redis-0",
          dtypes.int64,
          dtypes.int32,
          initializer=0,
          dim=8,
          kv_creator=de.RedisTableCreator(config=redis_config))
      table.clear()
      self.evaluate(table.size())

  def test_variable(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    id = 0
    if test_util.is_gpu_available():
      dim_list = [1, 2, 4, 8, 10, 16, 32, 64, 100, 200]
      kv_list = [[dtypes.int64, dtypes.float32], [dtypes.int64, dtypes.int32],
                 [dtypes.int64, dtypes.half], [dtypes.int64, dtypes.int8]]
    else:
      dim_list = [1, 8, 16, 128]
      kv_list = [[dtypes.int32, dtypes.double], [dtypes.int32, dtypes.float32],
                 [dtypes.int32, dtypes.int32], [dtypes.int64, dtypes.double],
                 [dtypes.int64, dtypes.float32], [dtypes.int64, dtypes.int32],
                 [dtypes.int64, dtypes.int64], [dtypes.int64, dtypes.string],
                 [dtypes.int64, dtypes.int8], [dtypes.int64, dtypes.half],
                 [dtypes.string, dtypes.double],
                 [dtypes.string, dtypes.float32], [dtypes.string, dtypes.int32],
                 [dtypes.string, dtypes.int64], [dtypes.string, dtypes.int8],
                 [dtypes.string, dtypes.half]]

    def _convert(v, t):
      return np.array(v).astype(_type_converter(t))

    for (key_dtype, value_dtype), dim in itertools.product(kv_list, dim_list):
      # Skip float16 tests if the platform is macOS arm64 architecture
      if is_macos() and is_arm64():
        if value_dtype == dtypes.half or value_dtype == dtypes.bfloat16:
          continue
      id += 1
      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        keys = constant_op.constant(
            np.array([0, 1, 2, 3]).astype(_type_converter(key_dtype)),
            key_dtype)
        values = constant_op.constant(
            _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype),
            value_dtype)
        table = de.get_variable(
            't1-' + str(id) + '_test_variable_' + key_dtype.name + '_' +
            value_dtype.name,
            key_dtype=key_dtype,
            value_dtype=value_dtype,
            initializer=np.array([-1]).astype(_type_converter(value_dtype)),
            dim=dim,
            kv_creator=de.RedisTableCreator(config=redis_config))

        table.clear()

        self.assertAllEqual(0, self.evaluate(table.size()))

        self.evaluate(table.upsert(keys, values))
        self.assertAllEqual(4, self.evaluate(table.size()))

        remove_keys = constant_op.constant(_convert([1, 5], key_dtype),
                                           key_dtype)
        self.evaluate(table.remove(remove_keys))
        self.assertAllEqual(3, self.evaluate(table.size()))

        remove_keys = constant_op.constant(_convert([0, 1, 5], key_dtype),
                                           key_dtype)
        output = table.lookup(remove_keys)
        self.assertAllEqual([3, dim], output.get_shape())

        result = self.evaluate(output)
        self.assertAllEqual(
            _convert([[0] * dim, [-1] * dim, [-1] * dim], value_dtype),
            _convert(result, value_dtype))

        exported_keys, exported_values = table.export()

        # exported data is in the order of the internal map, i.e. undefined
        sorted_keys = np.sort(self.evaluate(exported_keys))
        sorted_values = np.sort(self.evaluate(exported_values), axis=0)
        self.assertAllEqual(_convert([0, 2, 3], key_dtype),
                            _convert(sorted_keys, key_dtype))
        self.assertAllEqual(
            _convert([[0] * dim, [2] * dim, [3] * dim], value_dtype),
            _convert(sorted_values, value_dtype))

        table.clear()
        del table

  # def test_variable_find_with_exists_and_accum(self):
  #   id = 0
  #   if test_util.is_gpu_available():
  #     dim_list = [1, 2, 4, 8, 10, 16, 32, 64, 100, 200]
  #     kv_list = [[dtypes.int64, dtypes.float32], [dtypes.int64, dtypes.int32],
  #                [dtypes.int64, dtypes.half], [dtypes.int64, dtypes.int8]]
  #   else:
  #     dim_list = [1, 8, 16, 128]
  #     kv_list = [[dtypes.int32, dtypes.double], [dtypes.int32, dtypes.float32],
  #                [dtypes.int32, dtypes.int32], [dtypes.int64, dtypes.double],
  #                [dtypes.int64, dtypes.float32], [dtypes.int64, dtypes.int32],
  #                [dtypes.int64, dtypes.int64], [dtypes.int64, dtypes.int8],
  #                [dtypes.int64, dtypes.half], [dtypes.string, dtypes.double],
  #                [dtypes.string, dtypes.float32], [dtypes.string, dtypes.int32],
  #                [dtypes.string, dtypes.int64], [dtypes.string, dtypes.int8],
  #                [dtypes.string, dtypes.half]]

  #   def _convert(v, t):
  #     return np.array(v).astype(_type_converter(t))

  #   for (key_dtype, value_dtype), dim in itertools.product(kv_list, dim_list):
  #     id += 1
  #     with self.session(config=default_config,
  #                       use_gpu=test_util.is_gpu_available()) as sess:
  #       base_keys = constant_op.constant(
  #           np.array([0, 1, 2, 3]).astype(_type_converter(key_dtype)),
  #           key_dtype)
  #       base_values = constant_op.constant(
  #           _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype),
  #           value_dtype)

  #       simulate_other_process_add_keys = constant_op.constant(
  #           np.array([100]).astype(_type_converter(key_dtype)), key_dtype)
  #       simulate_other_process_add_vals = constant_op.constant(
  #           _convert([
  #               [99] * dim,
  #           ], value_dtype), value_dtype)

  #       simulate_other_process_remove_keys = constant_op.constant(
  #           np.array([1]).astype(_type_converter(key_dtype)), key_dtype)
  #       accum_keys = constant_op.constant(
  #           np.array([0, 1, 100, 3]).astype(_type_converter(key_dtype)),
  #           key_dtype)
  #       old_values = constant_op.constant(
  #           _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype),
  #           value_dtype)
  #       new_values = constant_op.constant(
  #           _convert([[10] * dim, [11] * dim, [100] * dim, [13] * dim],
  #                    value_dtype), value_dtype)
  #       exported_exists = constant_op.constant([True, True, False, True],
  #                                              dtype=dtypes.bool)

  #       table = de.get_variable('taccum1-' + str(id),
  #                               key_dtype=key_dtype,
  #                               value_dtype=value_dtype,
  #                               initializer=np.array([-1]).astype(
  #                                   _type_converter(value_dtype)),
  #                               dim=dim)
  #       self.evaluate(table.clear())

  #       self.assertAllEqual(0, self.evaluate(table.size()))

  #       self.evaluate(table.upsert(base_keys, base_values))
  #       _, exists = table.lookup(accum_keys, return_exists=True)
  #       self.assertAllEqual(self.evaluate(exported_exists),
  #                           self.evaluate(exists))
  #       # Simulate multi-process situation that other process operated table,
  #       # between lookup and accum in this process.
  #       self.evaluate(
  #           table.upsert(simulate_other_process_add_keys,
  #                        simulate_other_process_add_vals))
  #       self.evaluate(table.remove(simulate_other_process_remove_keys))
  #       self.assertAllEqual(4, self.evaluate(table.size()))
  #       self.evaluate(
  #           table.accum(accum_keys, old_values, new_values, exported_exists))

  #       exported_keys, exported_values = table.export()

  #       # exported data is in the order of the internal map, i.e. undefined
  #       sorted_keys = np.sort(self.evaluate(exported_keys), axis=0)
  #       sorted_values = np.sort(self.evaluate(exported_values), axis=0)
  #       self.assertAllEqual(
  #           np.sort(_convert([0, 2, 3, 100], key_dtype), axis=0),
  #           _convert(sorted_keys, key_dtype))
  #       self.assertAllEqual(
  #           _convert([[2] * dim, [10] * dim, [13] * dim, [99] * dim],
  #                    value_dtype), _convert(sorted_values, value_dtype))

  #       self.evaluate(table.clear())
  #       del table

  def test_variable_initializer(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    id = 0
    for initializer, target_mean, target_stddev in [
        (-1.0, -1.0, 0.0),
        (init_ops.random_normal_initializer(0.0, 0.01, seed=2), 0.0, 0.01),
    ]:
      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()):
        id += 1
        keys = constant_op.constant(list(range(2**16)), dtypes.int64)
        table = de.get_variable(
            "t1" + str(id) + '_test_variable_initializer',
            key_dtype=dtypes.int64,
            value_dtype=dtypes.float32,
            initializer=initializer,
            dim=10,
            kv_creator=de.RedisTableCreator(config=redis_config))
        table.clear()
        vals_op = table.lookup(keys)
        mean = self.evaluate(math_ops.reduce_mean(vals_op))
        stddev = self.evaluate(math_ops.reduce_std(vals_op))
        rtol = 2e-5
        atol = rtol
        self.assertAllClose(target_mean, mean, rtol, atol)
        self.assertAllClose(target_stddev, stddev, rtol, atol)
        table.clear()

  def test_save_restore(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    if context.executing_eagerly():
      self.skipTest('skip eager test when using legacy Saver.')
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(config=default_config, graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0.0], [1.0], [2.0]], dtypes.float32)
      table = de.Variable(
          key_dtype=dtypes.int64,
          value_dtype=dtypes.float32,
          initializer=-1.0,
          name="t1",
          dim=1,
      )
      table.clear()

      save = saver.Saver(var_list=[v0, v1, table])
      self.evaluate(variables.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(0, self.evaluate(table.size()))
      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, six.string_types)
      self.assertEqual(save_path, val)

      table.clear()
      del table

    with self.session(config=default_config, graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      table = de.Variable(
          name="t1",
          key_dtype=dtypes.int64,
          value_dtype=dtypes.float32,
          initializer=-1.0,
          dim=1,
          checkpoint=True,
      )
      table.clear()

      self.evaluate(
          table.upsert(
              constant_op.constant([0, 1], dtypes.int64),
              constant_op.constant([[12.0], [24.0]], dtypes.float32),
          ))
      size_op = table.size()
      self.assertAllEqual(2, self.evaluate(size_op))

      save = saver.Saver(var_list=[v0, v1, table])

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual([10.0], self.evaluate(v0))
      self.assertEqual([20.0], self.evaluate(v1))

      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([5, 0, 1, 2, 6], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([[-1.0], [0.0], [1.0], [2.0], [-1.0]],
                          self.evaluate(output))

      table.clear()
      del table

  def test_save_restore_only_table(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    if context.executing_eagerly():
      self.skipTest('skip eager test when using legacy Saver.')
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(
        config=default_config,
        graph=ops.Graph(),
        use_gpu=test_util.is_gpu_available(),
    ) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      default_val = -1
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)
      table = de.Variable(
          dtypes.int64,
          dtypes.int32,
          name="t1",
          initializer=default_val,
          checkpoint=True,
      )
      table.clear()

      save = saver.Saver([table])
      self.evaluate(variables.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(0, self.evaluate(table.size()))
      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, six.string_types)
      self.assertEqual(save_path, val)

      table.clear()
      del table

    with self.session(
        config=default_config,
        graph=ops.Graph(),
        use_gpu=test_util.is_gpu_available(),
    ) as sess:
      default_val = -1
      table = de.Variable(
          dtypes.int64,
          dtypes.int32,
          name="t1",
          initializer=default_val,
          checkpoint=True,
      )
      table.clear()

      self.evaluate(
          table.upsert(
              constant_op.constant([0, 2], dtypes.int64),
              constant_op.constant([[12], [24]], dtypes.int32),
          ))
      self.assertAllEqual(2, self.evaluate(table.size()))

      save = saver.Saver([table._tables[0]])

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.

      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 2, 3, 4], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([[0], [1], [2], [-1], [-1]], self.evaluate(output))

      table.clear()
      del table

  def test_training_save_restore(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    if context.executing_eagerly():
      self.skipTest('skip eager test when using legacy Saver.')
    opt = de.DynamicEmbeddingOptimizer(adam.AdamOptimizer(0.3))
    id = 0
    if test_util.is_gpu_available():
      dim_list = [1, 2, 4, 8, 10, 16, 32, 64, 100, 200]
    else:
      dim_list = [10]
    for key_dtype, value_dtype, dim, step in itertools.product(
        [dtypes.int64],
        [dtypes.float32],
        dim_list,
        [10],
    ):
      # Skip float16 tests if the platform is macOS arm64 architecture
      if is_macos() and is_arm64():
        if value_dtype == dtypes.half or value_dtype == dtypes.bfloat16:
          continue
      id += 1
      save_dir = os.path.join(self.get_temp_dir(), "save_restore")
      save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

      ids = script_ops.py_func(_create_dynamic_shape_tensor(),
                               inp=[],
                               Tout=key_dtype,
                               stateful=True)

      params = de.get_variable(
          name="params-test-0915-" + str(id) + '_test_training_save_restore',
          key_dtype=key_dtype,
          value_dtype=value_dtype,
          initializer=init_ops.random_normal_initializer(0.0, 0.01),
          dim=dim,
          kv_creator=de.RedisTableCreator(config=redis_config))
      params.clear()
      params_size = self.evaluate(params.size())

      _, var0 = de.embedding_lookup(params, ids, return_trainable=True)

      def loss():
        return var0 * var0

      params_keys, params_vals = params.export()
      mini = opt.minimize(loss, var_list=[var0])
      opt_slots = [opt.get_slot(var0, _s) for _s in opt.get_slot_names()]
      _saver = saver.Saver([params] + [_s.params for _s in opt_slots])

      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        self.evaluate(variables.global_variables_initializer())
        for _i in range(step):
          self.evaluate([mini])
        size_before_saved = self.evaluate(params.size())
        np_params_keys_before_saved = self.evaluate(params_keys)
        np_params_vals_before_saved = self.evaluate(params_vals)
        opt_slots_kv_pairs = [_s.params.export() for _s in opt_slots]
        np_slots_kv_pairs_before_saved = [
            self.evaluate(_kv) for _kv in opt_slots_kv_pairs
        ]
        params_size = self.evaluate(params.size())
        _saver.save(sess, save_path)

      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        self.evaluate(variables.global_variables_initializer())
        self.assertAllEqual(params_size, self.evaluate(params.size()))

        _saver.restore(sess, save_path)
        params_keys_restored, params_vals_restored = params.export()
        size_after_restored = self.evaluate(params.size())
        np_params_keys_after_restored = self.evaluate(params_keys_restored)
        np_params_vals_after_restored = self.evaluate(params_vals_restored)

        opt_slots_kv_pairs_restored = [_s.params.export() for _s in opt_slots]
        np_slots_kv_pairs_after_restored = [
            self.evaluate(_kv) for _kv in opt_slots_kv_pairs_restored
        ]
        self.assertAllEqual(size_before_saved, size_after_restored)
        self.assertAllEqual(
            np.sort(np_params_keys_before_saved),
            np.sort(np_params_keys_after_restored),
        )
        self.assertAllEqual(
            np.sort(np_params_vals_before_saved, axis=0),
            np.sort(np_params_vals_after_restored, axis=0),
        )
        for pairs_before, pairs_after in zip(np_slots_kv_pairs_before_saved,
                                             np_slots_kv_pairs_after_restored):
          self.assertAllEqual(
              np.sort(pairs_before[0], axis=0),
              np.sort(pairs_after[0], axis=0),
          )
          self.assertAllEqual(
              np.sort(pairs_before[1], axis=0),
              np.sort(pairs_after[1], axis=0),
          )
        if test_util.is_gpu_available():
          self.assertTrue("GPU" in params.tables[0].resource_handle.device)

  def test_training_save_restore_by_files(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    if context.executing_eagerly():
      self.skipTest('skip eager test when using legacy Saver.')
    opt = de.DynamicEmbeddingOptimizer(adam.AdamOptimizer(0.3))
    id = 0
    for key_dtype, value_dtype, dim, step in itertools.product(
        [dtypes.int64],
        [dtypes.float32],
        [10],
        [10],
    ):
      # Skip float16 tests if the platform is macOS arm64 architecture
      if is_macos() and is_arm64():
        if value_dtype == dtypes.half or value_dtype == dtypes.bfloat16:
          continue
      id += 1
      save_dir = os.path.join(self.get_temp_dir(), "save_restore")
      save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

      os.makedirs(save_path)
      redis_config_path = os.path.join(save_path, "redis_config_modify.json")
      redis_config_params_modify = {
          "redis_host_ip": ["127.0.0.1"],
          "redis_host_port": [6379],
          "table_store_mode": 1,
          "model_lib_abs_dir": save_path
      }
      with open(redis_config_path, 'w', encoding='utf-8') as f:
        f.write(
            json.dumps(redis_config_params_modify, indent=2, ensure_ascii=True))
      redis_config_modify = de.RedisTableConfig(
          redis_config_abs_dir=redis_config_path)

      ids = script_ops.py_func(_create_dynamic_shape_tensor(),
                               inp=[],
                               Tout=key_dtype,
                               stateful=True)

      params = de.get_variable(
          name="params-test-0916-" + str(id) +
          '_test_training_save_restore_by_files',
          key_dtype=key_dtype,
          value_dtype=value_dtype,
          initializer=0,
          dim=dim,
          kv_creator=de.RedisTableCreator(config=redis_config_modify))

      _, var0 = de.embedding_lookup(params, ids, return_trainable=True)

      def loss():
        return var0 * var0

      mini = opt.minimize(loss, var_list=[var0])
      opt_slots = [opt.get_slot(var0, _s) for _s in opt.get_slot_names()]
      _saver = saver.Saver([params] + [_s.params for _s in opt_slots])

      keys = np.random.randint(1, 100, dim)
      values = np.random.rand(keys.shape[0], dim)

      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(params.upsert(keys, values))
        params_vals = params.lookup(keys)
        for _i in range(step):
          self.evaluate([mini])
        size_before_saved = self.evaluate(params.size())
        np_params_vals_before_saved = self.evaluate(params_vals)
        params_size = self.evaluate(params.size())
        _saver.save(sess, save_path)

      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        _saver.restore(sess, save_path)
        self.evaluate(variables.global_variables_initializer())
        self.assertAllEqual(params_size, self.evaluate(params.size()))
        params_vals_restored = params.lookup(keys)
        size_after_restored = self.evaluate(params.size())
        np_params_vals_after_restored = self.evaluate(params_vals_restored)

        self.assertAllEqual(size_before_saved, size_after_restored)
        self.assertAllEqual(
            np.sort(np_params_vals_before_saved, axis=0),
            np.sort(np_params_vals_after_restored, axis=0),
        )

      params.clear()

  def test_get_variable(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    with self.session(
        config=default_config,
        graph=ops.Graph(),
        use_gpu=test_util.is_gpu_available(),
    ):
      default_val = -1
      with variable_scope.variable_scope("embedding", reuse=True):
        table1 = de.get_variable(
            "t1" + '_test_get_variable',
            dtypes.int64,
            dtypes.int32,
            initializer=default_val,
            dim=2,
            kv_creator=de.RedisTableCreator(config=redis_config))
        table2 = de.get_variable(
            "t1" + '_test_get_variable',
            dtypes.int64,
            dtypes.int32,
            initializer=default_val,
            dim=2,
            kv_creator=de.RedisTableCreator(config=redis_config))
        table3 = de.get_variable(
            "t3" + '_test_get_variable',
            dtypes.int64,
            dtypes.int32,
            initializer=default_val,
            dim=2,
            kv_creator=de.RedisTableCreator(config=redis_config))

        table1.clear()
        table2.clear()
        table3.clear()

      self.assertAllEqual(table1, table2)
      self.assertNotEqual(table1, table3)

  def test_get_variable_reuse_error(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    ops.disable_eager_execution()
    with self.session(
        config=default_config,
        graph=ops.Graph(),
        use_gpu=test_util.is_gpu_available(),
    ):
      with variable_scope.variable_scope("embedding", reuse=False):
        _ = de.get_variable(
            "t900",
            initializer=-1,
            dim=2,
            kv_creator=de.RedisTableCreator(config=redis_config))
        with self.assertRaisesRegexp(ValueError,
                                     "Variable embedding/t900 already exists"):
          _ = de.get_variable(
              "t900",
              initializer=-1,
              dim=2,
              kv_creator=de.RedisTableCreator(config=redis_config))

  @test_util.run_v1_only("Multiple sessions")
  def test_sharing_between_multi_sessions(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    ops.disable_eager_execution()
    # Start a server to store the table state
    server = server_lib.Server({"local0": ["localhost:0"]},
                               protocol="grpc",
                               start=True)
    # Create two sessions sharing the same state
    session1 = session.Session(server.target, config=default_config)
    session2 = session.Session(server.target, config=default_config)

    table = de.get_variable(
        "tx100" + '_test_sharing_between_multi_sessions',
        dtypes.int64,
        dtypes.int32,
        initializer=0,
        dim=1,
        kv_creator=de.RedisTableCreator(config=redis_config))
    self.evaluate(table.clear())

    # Populate the table in the first session
    with session1:
      with ops.device(_get_devices()[0]):
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(variables.local_variables_initializer())
        self.assertAllEqual(0, table.size().eval())

        keys = constant_op.constant([11, 12], dtypes.int64)
        values = constant_op.constant([[11], [12]], dtypes.int32)
        table.upsert(keys, values).run()
        self.assertAllEqual(2, table.size().eval())

        output = table.lookup(constant_op.constant([11, 12, 13], dtypes.int64))
        self.assertAllEqual([[11], [12], [0]], output.eval())

    # Verify that we can access the shared data from the second session
    with session2:
      with ops.device(_get_devices()[0]):
        self.assertAllEqual(2, table.size().eval())

        output = table.lookup(constant_op.constant([10, 11, 12], dtypes.int64))
        self.assertAllEqual([[0], [11], [12]], output.eval())

  def test_dynamic_embedding_variable(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()):
      default_val = constant_op.constant([-1, -2], dtypes.int64)
      keys = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      values = constant_op.constant([[0, 1], [2, 3], [4, 5], [6, 7]],
                                    dtypes.int32)
      table = de.get_variable(
          "t10" + '_test_dynamic_embedding_variable',
          dtypes.int64,
          dtypes.int32,
          initializer=default_val,
          dim=2,
          kv_creator=de.RedisTableCreator(config=redis_config))
      table.clear()

      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_keys = constant_op.constant([3, 4], dtypes.int64)
      self.evaluate(table.remove(remove_keys))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 4], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([3, 2], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual([[0, 1], [2, 3], [-1, -2]], result)

      exported_keys, exported_values = table.export()
      # exported data is in the order of the internal map, i.e. undefined
      sorted_keys = np.sort(self.evaluate(exported_keys))
      sorted_values = np.sort(self.evaluate(exported_values), axis=0)
      self.assertAllEqual([0, 1, 2], sorted_keys)
      sorted_expected_values = np.sort([[4, 5], [2, 3], [0, 1]], axis=0)
      self.assertAllEqual(sorted_expected_values, sorted_values)

      table.clear()
      del table

  def test_dynamic_embedding_variable_export_insert(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()):
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int32)
      table1 = de.get_variable(
          "t101" + '_test_dynamic_embedding_variable_export_insert',
          dtypes.int64,
          dtypes.int32,
          initializer=default_val,
          dim=2,
          kv_creator=de.RedisTableCreator(config=redis_config))

      table1.clear()

      self.assertAllEqual(0, self.evaluate(table1.size()))
      self.evaluate(table1.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table1.size()))

      input_keys = constant_op.constant([0, 1, 3], dtypes.int64)
      expected_output = [[0, 1], [2, 3], [-1, -1]]
      output1 = table1.lookup(input_keys)
      self.assertAllEqual(expected_output, self.evaluate(output1))

      exported_keys, exported_values = table1.export()
      self.assertAllEqual(3, self.evaluate(exported_keys).size)
      self.assertAllEqual(6, self.evaluate(exported_values).size)

      # Populate a second table from the exported data
      table2 = de.get_variable(
          "t102" + '_test_dynamic_embedding_variable_export_insert',
          dtypes.int64,
          dtypes.int32,
          initializer=default_val,
          dim=2,
          kv_creator=de.RedisTableCreator(config=redis_config))

      table2.clear()

      self.assertAllEqual(0, self.evaluate(table2.size()))
      self.evaluate(table2.upsert(exported_keys, exported_values))
      self.assertAllEqual(3, self.evaluate(table2.size()))

      # Verify lookup result is still the same
      output2 = table2.lookup(input_keys)
      self.assertAllEqual(expected_output, self.evaluate(output2))

  def test_dynamic_embedding_variable_invalid_shape(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()):
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      table = de.get_variable(
          "t110" + '_test_dynamic_embedding_variable_invalid_shape',
          dtypes.int64,
          dtypes.int32,
          initializer=default_val,
          dim=2,
          kv_creator=de.RedisTableCreator(config=redis_config))

      table.clear()

      # Shape [6] instead of [3, 2]
      values = constant_op.constant([0, 1, 2, 3, 4, 5], dtypes.int32)
      with self.assertRaisesOpError("Expected shape"):
        self.evaluate(table.upsert(keys, values))

      # Shape [2,3] instead of [3, 2]
      values = constant_op.constant([[0, 1, 2], [3, 4, 5]], dtypes.int32)
      with self.assertRaisesOpError("Expected shape"):
        self.evaluate(table.upsert(keys, values))

      # Shape [2, 2] instead of [3, 2]
      values = constant_op.constant([[0, 1], [2, 3]], dtypes.int32)
      with self.assertRaisesOpError("Expected shape"):
        self.evaluate(table.upsert(keys, values))

      # Shape [3, 1] instead of [3, 2]
      values = constant_op.constant([[0], [2], [4]], dtypes.int32)
      with self.assertRaisesOpError("Expected shape"):
        self.evaluate(table.upsert(keys, values))

      # Valid Insert
      values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int32)
      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

  def test_dynamic_embedding_variable_duplicate_insert(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([0, 1, 2, 2], dtypes.int64)
      values = constant_op.constant([[0.0], [1.0], [2.0], [3.0]],
                                    dtypes.float32)
      table = de.get_variable(
          "t130" + '_test_dynamic_embedding_variable_duplicate_insert',
          dtypes.int64,
          dtypes.float32,
          initializer=default_val,
          kv_creator=de.RedisTableCreator(config=redis_config))

      table.clear()

      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_keys = constant_op.constant([0, 1, 2], dtypes.int64)
      output = table.lookup(input_keys)

      result = self.evaluate(output)
      self.assertTrue(
          list(result) in [[[0.0], [1.0], [3.0]], [[0.0], [1.0], [2.0]]])

  def test_dynamic_embedding_variable_find_high_rank(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)
      table = de.get_variable(
          "t140" + '_test_dynamic_embedding_variable_find_high_rank',
          dtypes.int64,
          dtypes.int32,
          initializer=default_val,
          kv_creator=de.RedisTableCreator(config=redis_config))

      table.clear()

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_keys = constant_op.constant([[0, 1], [2, 4]], dtypes.int64)
      output = table.lookup(input_keys)
      self.assertAllEqual([2, 2, 1], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual([[[0], [1]], [[2], [-1]]], result)

  def test_dynamic_embedding_variable_insert_low_rank(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
      values = constant_op.constant([[[0], [1]], [[2], [3]]], dtypes.int32)
      table = de.get_variable(
          "t150" + '_test_dynamic_embedding_variable_insert_low_rank',
          dtypes.int64,
          dtypes.int32,
          initializer=default_val,
          kv_creator=de.RedisTableCreator(config=redis_config))

      table.clear()

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 3, 4], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertAllEqual([[0], [1], [3], [-1]], result)

  def test_dynamic_embedding_variable_remove_low_rank(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
      values = constant_op.constant([[[0], [1]], [[2], [3]]], dtypes.int32)
      table = de.get_variable(
          "t160" + '_test_dynamic_embedding_variable_remove_low_rank',
          dtypes.int64,
          dtypes.int32,
          initializer=default_val,
          kv_creator=de.RedisTableCreator(config=redis_config))

      table.clear()

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_keys = constant_op.constant([1, 4], dtypes.int64)
      self.evaluate(table.remove(remove_keys))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 3, 4], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertAllEqual([[0], [-1], [3], [-1]], result)

  def test_dynamic_embedding_variable_insert_high_rank(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = constant_op.constant([-1, -1, -1], dtypes.int32)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                                    dtypes.int32)
      table = de.get_variable(
          "t170" + '_test_dynamic_embedding_variable_insert_high_rank',
          dtypes.int64,
          dtypes.int32,
          initializer=default_val,
          dim=3,
          kv_creator=de.RedisTableCreator(config=redis_config))

      table.clear()

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([[0, 1], [3, 4]], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([2, 2, 3], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual(
          [[[0, 1, 2], [2, 3, 4]], [[-1, -1, -1], [-1, -1, -1]]], result)

  def test_dynamic_embedding_variable_remove_high_rank(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = constant_op.constant([-1, -1, -1], dtypes.int32)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                                    dtypes.int32)
      table = de.get_variable(
          "t180" + '_test_dynamic_embedding_variable_remove_high_rank',
          dtypes.int64,
          dtypes.int32,
          initializer=default_val,
          dim=3,
          kv_creator=de.RedisTableCreator(config=redis_config))

      table.clear()

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([[0, 3]], dtypes.int64)
      self.evaluate(table.remove(remove_keys))
      self.assertAllEqual(2, self.evaluate(table.size()))

      remove_keys = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([2, 2, 3], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual(
          [[[-1, -1, -1], [2, 3, 4]], [[4, 5, 6], [-1, -1, -1]]], result)

  def test_dynamic_embedding_variables(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)

      table1 = de.get_variable(
          "t191" + '_test_dynamic_embedding_variables',
          dtypes.int64,
          dtypes.int32,
          initializer=default_val,
          kv_creator=de.RedisTableCreator(config=redis_config))
      table2 = de.get_variable(
          "t192" + '_test_dynamic_embedding_variables',
          dtypes.int64,
          dtypes.int32,
          initializer=default_val,
          kv_creator=de.RedisTableCreator(config=redis_config))
      table3 = de.get_variable(
          "t193" + '_test_dynamic_embedding_variables',
          dtypes.int64,
          dtypes.int32,
          initializer=default_val,
          kv_creator=de.RedisTableCreator(config=redis_config))

      table1.clear()
      table2.clear()
      table3.clear()

      self.evaluate(table1.upsert(keys, values))
      self.evaluate(table2.upsert(keys, values))
      self.evaluate(table3.upsert(keys, values))

      self.assertAllEqual(3, self.evaluate(table1.size()))
      self.assertAllEqual(3, self.evaluate(table2.size()))
      self.assertAllEqual(3, self.evaluate(table3.size()))

      remove_keys = constant_op.constant([0, 1, 3], dtypes.int64)
      output1 = table1.lookup(remove_keys)
      output2 = table2.lookup(remove_keys)
      output3 = table3.lookup(remove_keys)

      out1, out2, out3 = self.evaluate([output1, output2, output3])
      self.assertAllEqual([[0], [1], [-1]], out1)
      self.assertAllEqual([[0], [1], [-1]], out2)
      self.assertAllEqual([[0], [1], [-1]], out3)

  def test_dynamic_embedding_variable_with_tensor_default(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = constant_op.constant(-1, dtypes.int32)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)
      table = de.get_variable(
          "t200" + '_test_dynamic_embedding_variable_with_tensor_default',
          dtypes.int64,
          dtypes.int32,
          initializer=default_val,
          kv_creator=de.RedisTableCreator(config=redis_config))

      table.clear()

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 3], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertAllEqual([[0], [1], [-1]], result)

  def test_signature_mismatch(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    with self.session(config=config, use_gpu=test_util.is_gpu_available()):
      default_val = -1
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)
      table = de.get_variable(
          "t210" + '_test_signature_mismatch',
          dtypes.int64,
          dtypes.int32,
          initializer=default_val,
          kv_creator=de.RedisTableCreator(config=redis_config))

      table.clear()

      # upsert with keys of the wrong type
      with self.assertRaises(ValueError):
        self.evaluate(
            table.upsert(constant_op.constant([4.0, 5.0, 6.0], dtypes.float32),
                         values))

      # upsert with values of the wrong type
      with self.assertRaises(ValueError):
        self.evaluate(table.upsert(keys, constant_op.constant(["a", "b", "c"])))

      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys_ref = variables.Variable(0, dtype=dtypes.int64)
      input_int64_ref = variables.Variable([-1], dtype=dtypes.int32)
      self.evaluate(variables.global_variables_initializer())

      # Ref types do not produce an upsert signature mismatch.
      self.evaluate(table.upsert(remove_keys_ref, input_int64_ref))
      self.assertAllEqual(3, self.evaluate(table.size()))

      # Ref types do not produce a lookup signature mismatch.
      self.assertEqual([-1], self.evaluate(table.lookup(remove_keys_ref)))

      # lookup with keys of the wrong type
      remove_keys = constant_op.constant([1, 2, 3], dtypes.int32)
      with self.assertRaises(ValueError):
        self.evaluate(table.lookup(remove_keys))

  def test_dynamic_embedding_variable_int_float(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()):
      default_val = -1.0
      keys = constant_op.constant([3, 7, 0], dtypes.int64)
      values = constant_op.constant([[7.5], [-1.2], [9.9]], dtypes.float32)
      table = de.get_variable(
          "t220" + '_test_dynamic_embedding_variable_int_float',
          dtypes.int64,
          dtypes.float32,
          initializer=default_val,
          kv_creator=de.RedisTableCreator(config=redis_config))

      table.clear()

      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([7, 0, 11], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertAllClose([[-1.2], [9.9], [default_val]], result)

  def test_dynamic_embedding_variable_with_random_init(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0.0], [1.0], [2.0]], dtypes.float32)
      default_val = init_ops.random_uniform_initializer()
      table = de.get_variable(
          "t230" + '_test_dynamic_embedding_variable_with_random_init',
          dtypes.int64,
          dtypes.float32,
          initializer=default_val,
          kv_creator=de.RedisTableCreator(config=redis_config))

      table.clear()

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 3], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertNotEqual([-1.0], result[2])

  def test_dynamic_embedding_variable_with_restrict_v1(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    if context.executing_eagerly():
      self.skipTest('skip eager test when using legacy optimizers.')

    optmz = de.DynamicEmbeddingOptimizer(adam.AdamOptimizer(0.1))
    data_len = 32
    maxval = 256
    num_reserved = 100
    trigger = 150
    embed_dim = 8

    var_guard_by_tstp = de.get_variable(
        'tstp_guard' + '_test_dynamic_embedding_variable_with_restrict_v1',
        key_dtype=dtypes.int64,
        value_dtype=dtypes.float32,
        initializer=-1.,
        dim=embed_dim,
        init_size=256,
        restrict_policy=de.TimestampRestrictPolicy,
        kv_creator=de.RedisTableCreator(config=redis_config))

    var_guard_by_tstp.clear()

    var_guard_by_freq = de.get_variable(
        'freq_guard' + '_test_dynamic_embedding_variable_with_restrict_v1',
        key_dtype=dtypes.int64,
        value_dtype=dtypes.float32,
        initializer=-1.,
        dim=embed_dim,
        init_size=256,
        restrict_policy=de.FrequencyRestrictPolicy,
        kv_creator=de.RedisTableCreator(config=redis_config))

    var_guard_by_freq.clear()

    sparse_vars = [var_guard_by_tstp, var_guard_by_freq]

    indices = [data_fn((data_len, 1), maxval) for _ in range(3)]
    _, trainables, loss = model_fn(sparse_vars, embed_dim, indices)
    train_op = optmz.minimize(loss, var_list=trainables)

    var_sizes = [0, 0]
    self.evaluate(variables.global_variables_initializer())

    while not all(sz > trigger for sz in var_sizes):
      self.evaluate(train_op)
      var_sizes = self.evaluate([spv.size() for spv in sparse_vars])

    self.assertTrue(all(sz >= trigger for sz in var_sizes))
    tstp_restrict_op = var_guard_by_tstp.restrict(num_reserved, trigger=trigger)
    if tstp_restrict_op != None:
      self.evaluate(tstp_restrict_op)
    freq_restrict_op = var_guard_by_freq.restrict(num_reserved, trigger=trigger)
    if freq_restrict_op != None:
      self.evaluate(freq_restrict_op)
    var_sizes = self.evaluate([spv.size() for spv in sparse_vars])
    self.assertAllEqual(var_sizes, [num_reserved, num_reserved])

    slot_params = []
    for _trainable in trainables:
      slot_params += [
          optmz.get_slot(_trainable, name).params
          for name in optmz.get_slot_names()
      ]
    slot_params = list(set(slot_params))

    for sp in slot_params:
      self.assertAllEqual(self.evaluate(sp.size()), num_reserved)
    tstp_size = self.evaluate(var_guard_by_tstp.restrict_policy.status.size())
    self.assertAllEqual(tstp_size, num_reserved)
    freq_size = self.evaluate(var_guard_by_freq.restrict_policy.status.size())
    self.assertAllEqual(freq_size, num_reserved)

  def test_dynamic_embedding_variable_with_restrict_v2(self):
    if _redis_health_check(redis_config_params["redis_host_ip"][0],
                           redis_config_params["redis_host_port"][0]) == False:
      self.skipTest('skip redis test when unable to access the redis service.')
    if not context.executing_eagerly():
      self.skipTest('Test in eager mode only.')

    optmz = de.DynamicEmbeddingOptimizer(optimizer_v2.adam.Adam(0.1))
    data_len = 32
    maxval = 256
    num_reserved = 100
    trigger = 150
    embed_dim = 8
    trainables = []

    var_guard_by_tstp = de.get_variable(
        'tstp_guard' + '_test_dynamic_embedding_variable_with_restrict_v2',
        key_dtype=dtypes.int64,
        value_dtype=dtypes.float32,
        initializer=-1.,
        dim=embed_dim,
        restrict_policy=de.TimestampRestrictPolicy,
        kv_creator=de.RedisTableCreator(config=redis_config))

    var_guard_by_tstp.clear()

    var_guard_by_freq = de.get_variable(
        'freq_guard' + '_test_dynamic_embedding_variable_with_restrict_v2',
        key_dtype=dtypes.int64,
        value_dtype=dtypes.float32,
        initializer=-1.,
        dim=embed_dim,
        restrict_policy=de.FrequencyRestrictPolicy,
        kv_creator=de.RedisTableCreator(config=redis_config))

    var_guard_by_freq.clear()

    sparse_vars = [var_guard_by_tstp, var_guard_by_freq]

    def loss_fn(sparse_vars, trainables):
      indices = [data_fn((data_len, 1), maxval) for _ in range(3)]
      _, tws, loss = model_fn(sparse_vars, embed_dim, indices)
      trainables.clear()
      trainables.extend(tws)
      return loss

    def var_fn():
      return trainables

    var_sizes = [0, 0]

    while not all(sz > trigger for sz in var_sizes):
      optmz.minimize(lambda: loss_fn(sparse_vars, trainables), var_fn)
      var_sizes = [spv.size() for spv in sparse_vars]

    self.assertTrue(all(sz >= trigger for sz in var_sizes))
    var_guard_by_tstp.restrict(num_reserved, trigger=trigger)
    var_guard_by_freq.restrict(num_reserved, trigger=trigger)
    var_sizes = [spv.size() for spv in sparse_vars]
    self.assertAllEqual(var_sizes, [num_reserved, num_reserved])

    slot_params = []
    for _trainable in trainables:
      slot_params += [
          optmz.get_slot(_trainable, name).params
          for name in optmz.get_slot_names()
      ]
    slot_params = list(set(slot_params))

    for sp in slot_params:
      self.assertAllEqual(sp.size(), num_reserved)
    self.assertAllEqual(var_guard_by_tstp.restrict_policy.status.size(),
                        num_reserved)
    self.assertAllEqual(var_guard_by_freq.restrict_policy.status.size(),
                        num_reserved)


if __name__ == "__main__":
  if is_windows() == False:
    if os.popen("which redis-server").read() == '':
      print("redis-server is not exist, now try to install redis-server.")
      if is_macos():
        print(os.popen("brew install -y redis > /dev/null 2> /dev/null").read())
      elif is_linux():
        if os.popen("which apt").read() != '':
          print(
              os.popen("apt install -y redis > /dev/null 2> /dev/null").read())
        elif os.popen("which yum").read() != '':
          print(
              os.popen("yum install -y redis > /dev/null 2> /dev/null").read())
        else:
          print(
              "Unknow Linux release version, please install Redis by yourself.")
          sys.exit()
      elif is_raspi_arm():
        if os.popen("which apt").read() != '':
          print(
              os.popen("apt install -y redis > /dev/null 2> /dev/null").read())
        elif os.popen("which yum").read() != '':
          print(
              os.popen("yum install -y redis > /dev/null 2> /dev/null").read())
        else:
          print(
              "Unknow ARM system release version, please install Redis by yourself."
          )
          sys.exit()
      else:
        print("Unknow platform system, please install Redis by yourself.")
        sys.exit()
  elif is_windows():
    if os.popen("where redis-server").read() == '':
      if os.popen("where winget").read() != '':
        print(os.popen("winget install redis").read())
      else:
        print("Unsupported Windows version, please install Redis by yourself.")
        sys.exit()

  print(os.popen("redis-server --port 6379 --daemonize yes").read())
  print(os.popen("redis-cli -p 6379 INFO").read())
  print(
      os.popen('redis-cli -h ' + redis_config_params["redis_host_ip"][0] +
               ' -p ' + str(redis_config_params["redis_host_port"][0]) +
               ' FLUSHALL').read())
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  if _redis_health_check(redis_config_params["redis_host_ip"][0],
                         redis_config_params["redis_host_port"][0]) == False:
    exit(f"skip {__file__} redis test when unable to access the redis service.")
  test.main()
