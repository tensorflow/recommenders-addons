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
import math
import numpy as np
import os
from packaging import version
import six
import tempfile

from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.utils.check_platform import is_macos, is_arm64

from tensorflow import version as tf_version
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.keras import initializers as kinit
from tensorflow.python.keras import optimizer_v2
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.training import training
if version.parse(tf_version.VERSION) >= version.parse("2.14"):
  from tensorflow.python.checkpoint.checkpoint import Checkpoint
else:
  from tensorflow.python.training.tracking.util import Checkpoint
from tensorflow.python.util import compat

if version.parse(tf_version.VERSION) >= version.parse("2.16"):
  try:  # independently import tf_keras
    from tf_keras import layers
  except:
    from tensorflow.python.keras import layers
else:
  from tensorflow_estimator.python.estimator import estimator
  from tensorflow_estimator.python.estimator import estimator_lib
  from tensorflow.keras import layers

if version.parse(tf_version.VERSION) >= version.parse("2.11"):
  # The data_structures has been moved to the new package in tf 2.11
  from tensorflow.python.trackable import data_structures
else:
  from tensorflow.python.training.tracking import data_structures

try:
  import tensorflow_io
except:
  print()


# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
def _type_converter(tf_type):
  mapper = {
      dtypes.int32: np.int32,
      dtypes.int64: np.int64,
      dtypes.float32: float,
      dtypes.float64: np.float64,
      dtypes.string: str,
      dtypes.half: np.float16,
      dtypes.bfloat16: dtypes.bfloat16.as_numpy_dtype,
      dtypes.int8: np.int8,
      dtypes.bool: bool,
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


class TestGraph(object):

  def __init__(self,
               key_dtype,
               value_dtype,
               dim,
               num_shards,
               var_name,
               devar_name,
               run_id,
               x=None):
    self.key_dtype = key_dtype
    self.value_dtype = value_dtype
    self.dim = dim

    # common define
    init_ids = [0, 1, 2]
    self.init_vals = np.random.rand(3, self.dim)
    raw_ids = [1]
    if x is None:
      self.x = constant_op.constant(np.random.rand(self.dim, len(raw_ids)),
                                    dtype=self.value_dtype)
    else:
      self.x = ops.convert_to_tensor(x, dtype=self.value_dtype)

    # variable graph
    self.var = resource_variable_ops.ResourceVariable(
        name='t2021-' + var_name + str(run_id),
        initial_value=self.init_vals,
        dtype=self.value_dtype)
    ids = constant_op.constant(raw_ids, dtype=self.key_dtype)
    self.var_lookup = embedding_ops.embedding_lookup([self.var], ids)
    self.var_pred = math_ops.matmul(self.var_lookup, self.x)
    self.var_loss = self.var_pred * self.var_pred
    self.var_opt_op = adam.AdamOptimizer(0.1).minimize(self.var_loss)

    # deo variable graph
    self.devar = de.get_variable(name='t2021-' + devar_name + str(run_id),
                                 key_dtype=self.key_dtype,
                                 value_dtype=self.value_dtype,
                                 devices=_get_devices() * num_shards,
                                 initializer=1.,
                                 dim=dim)
    self.devar_init_op = self.devar.upsert(
        constant_op.constant(init_ids, dtype=self.key_dtype),
        constant_op.constant(self.init_vals, dtype=self.value_dtype))
    self.devar_lookup, self.detw = de.embedding_lookup([self.devar],
                                                       ids,
                                                       return_trainable=True)
    self.devar_pred = math_ops.matmul(self.devar_lookup, self.x)
    self.devar_loss = self.devar_pred * self.devar_pred
    optmz = de.DynamicEmbeddingOptimizer(adam.AdamOptimizer(0.1))
    self.devar_opt_op = optmz.minimize(self.devar_loss)
    self.check_de_keys, self.check_de_values = self.devar.export()


def _write_checkpoint(test, session):
  save = saver.Saver()
  ckpt_prefix = os.path.join(test.get_temp_dir(), 'model')
  save.save(session, ckpt_prefix, global_step=0)


def _get_meta_file(ckpt_dir):
  for fname in os.listdir(ckpt_dir):
    if fname.endswith(".meta"):
      return os.path.join(ckpt_dir, fname)
  else:
    raise ValueError("No meta file found in {}.".format(ckpt_dir))


default_config = config_pb2.ConfigProto(
    allow_soft_placement=True,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))


@test_util.run_all_in_graph_and_eager_modes
class VariableTest(test.TestCase):

  def test_variable(self):
    id = 0
    if test_util.is_gpu_available():
      dim_list = [1, 2, 4, 8, 10, 16, 32, 64, 100, 200]
      kv_list = [[dtypes.int64, dtypes.float32], [dtypes.int64, dtypes.int32],
                 [dtypes.int64, dtypes.half], [dtypes.int64, dtypes.int8],
                 [dtypes.int64, dtypes.int64], [dtypes.int64, dtypes.bfloat16]]
    else:
      dim_list = [1, 8, 16, 128]
      kv_list = [[dtypes.int32, dtypes.double], [dtypes.int32, dtypes.float32],
                 [dtypes.int32, dtypes.int32], [dtypes.int64, dtypes.double],
                 [dtypes.int64, dtypes.float32], [dtypes.int64, dtypes.int32],
                 [dtypes.int64, dtypes.int64], [dtypes.int64, dtypes.string],
                 [dtypes.int64, dtypes.int8], [dtypes.int64, dtypes.half],
                 [dtypes.int64,
                  dtypes.bfloat16], [dtypes.string, dtypes.double],
                 [dtypes.string, dtypes.float32], [dtypes.string, dtypes.int32],
                 [dtypes.string, dtypes.int64], [dtypes.string, dtypes.int8],
                 [dtypes.string, dtypes.half]]

    def _convert(v, t):
      return np.array(v).astype(_type_converter(t))

    for (key_dtype, value_dtype), dim in itertools.product(kv_list, dim_list):
      id += 1
      # Skip float16 and bfloat16 tests if the platform is macOS arm64 architecture
      if is_macos() and is_arm64():
        if value_dtype == dtypes.half or value_dtype == dtypes.bfloat16:
          continue
      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        keys = constant_op.constant(
            np.array([0, 1, 2, 3]).astype(_type_converter(key_dtype)),
            key_dtype)
        values = constant_op.constant(
            _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype),
            value_dtype)
        table = de.get_variable('t1-' + str(id),
                                key_dtype=key_dtype,
                                value_dtype=value_dtype,
                                initializer=np.array([-1]).astype(
                                    _type_converter(value_dtype)),
                                dim=dim)
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

        del table

  def test_variable_find_with_exists_and_accum(self):
    id = 0
    if test_util.is_gpu_available():
      dim_list = [1, 2, 4, 8, 10, 16, 32, 64, 100, 200]
      kv_list = [[dtypes.int64, dtypes.float32], [dtypes.int64, dtypes.int32],
                 [dtypes.int64, dtypes.half], [dtypes.int64, dtypes.int8]]
    else:
      dim_list = [1, 8, 16, 128]
      kv_list = [[dtypes.int32, dtypes.double], [dtypes.int32, dtypes.float32],
                 [dtypes.int32, dtypes.int32], [dtypes.int32, dtypes.bfloat16],
                 [dtypes.int64, dtypes.double], [dtypes.int64, dtypes.float32],
                 [dtypes.int64, dtypes.int32], [dtypes.int64, dtypes.int64],
                 [dtypes.int64, dtypes.int8], [dtypes.int64, dtypes.half],
                 [dtypes.int64,
                  dtypes.bfloat16], [dtypes.string, dtypes.double],
                 [dtypes.string, dtypes.float32], [dtypes.string, dtypes.int32],
                 [dtypes.string, dtypes.int64], [dtypes.string, dtypes.int8],
                 [dtypes.string, dtypes.half]]

    def _convert(v, t):
      return np.array(v).astype(_type_converter(t))

    for (key_dtype, value_dtype), dim in itertools.product(kv_list, dim_list):
      id += 1
      # Skip float16 and bfloat16 tests if the platform is macOS arm64 archtecture
      if is_macos() and is_arm64():
        if value_dtype == dtypes.half or value_dtype == dtypes.bfloat16:
          continue
      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        base_keys = constant_op.constant(
            np.array([0, 1, 2, 3]).astype(_type_converter(key_dtype)),
            key_dtype)
        base_values = constant_op.constant(
            _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype),
            value_dtype)

        simulate_other_process_add_keys = constant_op.constant(
            np.array([100]).astype(_type_converter(key_dtype)), key_dtype)
        simulate_other_process_add_vals = constant_op.constant(
            _convert([
                [99] * dim,
            ], value_dtype), value_dtype)

        simulate_other_process_remove_keys = constant_op.constant(
            np.array([1]).astype(_type_converter(key_dtype)), key_dtype)
        accum_keys = constant_op.constant(
            np.array([0, 1, 100, 3]).astype(_type_converter(key_dtype)),
            key_dtype)
        old_values = constant_op.constant(
            _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype),
            value_dtype)
        new_values = constant_op.constant(
            _convert([[10] * dim, [11] * dim, [100] * dim, [13] * dim],
                     value_dtype), value_dtype)
        exported_exists = constant_op.constant([True, True, False, True],
                                               dtype=dtypes.bool)

        table = de.get_variable('taccum1-' + str(id),
                                key_dtype=key_dtype,
                                value_dtype=value_dtype,
                                initializer=np.array([-1]).astype(
                                    _type_converter(value_dtype)),
                                dim=dim)

        self.assertAllEqual(0, self.evaluate(table.size()))

        self.evaluate(table.upsert(base_keys, base_values))
        _, exists = table.lookup(accum_keys, return_exists=True)
        self.assertAllEqual(self.evaluate(exported_exists),
                            self.evaluate(exists))
        # Simulate multi-process situation that other process operated table,
        # between lookup and accum in this process.
        self.evaluate(
            table.upsert(simulate_other_process_add_keys,
                         simulate_other_process_add_vals))
        self.evaluate(table.remove(simulate_other_process_remove_keys))
        self.assertAllEqual(4, self.evaluate(table.size()))
        self.evaluate(
            table.accum(accum_keys, old_values, new_values, exported_exists))

        exported_keys, exported_values = table.export()

        # exported data is in the order of the internal map, i.e. undefined
        sorted_keys = np.sort(self.evaluate(exported_keys), axis=0)
        sorted_values = np.sort(self.evaluate(exported_values), axis=0)
        self.assertAllEqual(
            np.sort(_convert([0, 2, 3, 100], key_dtype), axis=0),
            _convert(sorted_keys, key_dtype))
        self.assertAllEqual(
            _convert([[2] * dim, [10] * dim, [13] * dim, [99] * dim],
                     value_dtype), _convert(sorted_values, value_dtype))

        del table

  def test_variable_initializer(self):
    id = 0
    for initializer, target_mean, target_stddev in [
        (-1.0, -1.0, 0.0),
        (init_ops.random_normal_initializer(0.0, 0.01, seed=2), 0.0, 0.01),
    ]:
      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()):
        id += 1
        keys = constant_op.constant(list(range(2**17)), dtypes.int64)
        table = de.get_variable(
            "t1" + str(id),
            key_dtype=dtypes.int64,
            value_dtype=dtypes.float32,
            initializer=initializer,
            dim=10,
        )
        vals_op = table.lookup(keys)
        mean = self.evaluate(math_ops.reduce_mean(vals_op))
        stddev = self.evaluate(math_ops.reduce_std(vals_op))
        rtol = 2e-5
        atol = rtol
        self.assertAllClose(target_mean, mean, rtol, atol)
        self.assertAllClose(target_stddev, stddev, rtol, atol)

  def test_save_restore(self):
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

      del table

  def test_save_restore_file_system(self):
    if context.executing_eagerly():
      self.skipTest('skip eager test when using legacy Saver.')
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(config=default_config, graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0.0], [1.0], [2.0]], dtypes.float32)

      num_shards = 2
      table = de.Variable(
          key_dtype=dtypes.int64,
          value_dtype=dtypes.float32,
          initializer=-1.0,
          devices=_get_devices() * num_shards,
          name="t1",
          dim=1,
          kv_creator=de.CuckooHashTableCreator(saver=de.FileSystemSaver()))

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

      del table

    with self.session(config=default_config, graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      num_shards = 2
      table = de.Variable(
          key_dtype=dtypes.int64,
          value_dtype=dtypes.float32,
          initializer=-1.0,
          devices=_get_devices() * num_shards,
          name="t1",
          dim=1,
          checkpoint=True,
          kv_creator=de.CuckooHashTableCreator(saver=de.FileSystemSaver()))
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

      del table

    # test expand shards number
    with self.session(config=default_config, graph=ops.Graph()) as sess:
      num_shards = 3
      table = de.Variable(
          key_dtype=dtypes.int64,
          value_dtype=dtypes.float32,
          initializer=-1.0,
          devices=_get_devices() * num_shards,
          name="t1",
          dim=1,
          checkpoint=True,
          kv_creator=de.CuckooHashTableCreator(saver=de.FileSystemSaver()))
      self.evaluate(
          table.upsert(
              constant_op.constant([0, 1], dtypes.int64),
              constant_op.constant([[12.0], [24.0]], dtypes.float32),
          ))
      size_op = table.size()
      self.assertAllEqual(2, self.evaluate(size_op))

      save = saver.Saver(var_list=[table])

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([5, 0, 1, 2, 6], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([[-1.0], [0.0], [1.0], [2.0], [-1.0]],
                          self.evaluate(output))

      del table

    # test contracte shards number
    with self.session(config=default_config, graph=ops.Graph()) as sess:
      num_shards = 1
      table = de.Variable(
          key_dtype=dtypes.int64,
          value_dtype=dtypes.float32,
          initializer=-1.0,
          devices=_get_devices() * num_shards,
          name="t1",
          dim=1,
          checkpoint=True,
          kv_creator=de.CuckooHashTableCreator(saver=de.FileSystemSaver()))
      self.evaluate(
          table.upsert(
              constant_op.constant([0, 1], dtypes.int64),
              constant_op.constant([[12.0], [24.0]], dtypes.float32),
          ))
      size_op = table.size()
      self.assertAllEqual(2, self.evaluate(size_op))

      save = saver.Saver(var_list=[table])

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([5, 0, 1, 2, 6], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([[-1.0], [0.0], [1.0], [2.0], [-1.0]],
                          self.evaluate(output))

      del table

  def test_table_save_load_file_system(self):
    self.skipTest('Only test for file_system export, need file_system path.')
    if context.executing_eagerly():
      self.skipTest('skip eager test when using legacy Saver.')
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    os.environ["AWS_ACCESS_KEY_ID"] = "Q3AM3UQ867SPQQA43P2F"
    os.environ[
        "AWS_SECRET_ACCESS_KEY"] = "zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG"
    os.environ["S3_ENDPOINT"] = "https://play.min.io"

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

      save = saver.Saver(var_list=[v0, v1])
      self.evaluate(variables.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(0, self.evaluate(table.size()))
      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      # save table
      for k, v in enumerate(table.tables):
        self.evaluate(
            v.save_to_file_system("s3://test/" + str(k), buffer_size=4096))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, six.string_types)
      self.assertEqual(save_path, val)

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
      self.evaluate(
          table.upsert(
              constant_op.constant([0, 1], dtypes.int64),
              constant_op.constant([[12.0], [24.0]], dtypes.float32),
          ))
      size_op = table.size()
      self.assertAllEqual(2, self.evaluate(size_op))

      save = saver.Saver(var_list=[v0, v1])

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # load table
      for k, v in enumerate(table.tables):
        self.evaluate(
            v.load_from_file_system("s3://test/" + str(k), buffer_size=4096))
      # Check that the parameter nodes have been restored.
      self.assertEqual([10.0], self.evaluate(v0))
      self.assertEqual([20.0], self.evaluate(v1))

      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([5, 0, 1, 2, 6], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([[-1.0], [0.0], [1.0], [2.0], [-1.0]],
                          self.evaluate(output))

      del table

  def test_table_save_load_local_file_system(self):
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

      save = saver.Saver(var_list=[v0, v1])
      self.evaluate(variables.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(0, self.evaluate(table.size()))
      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      # save table
      for k, v in enumerate(table.tables):
        self.evaluate(
            v.save_to_file_system("file:///tmp/test_local_file_system/" +
                                  str(k),
                                  buffer_size=4096))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, six.string_types)
      self.assertEqual(save_path, val)

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
      self.evaluate(table.clear())
      self.evaluate(
          table.upsert(
              constant_op.constant([0, 1], dtypes.int64),
              constant_op.constant([[12.0], [24.0]], dtypes.float32),
          ))
      size_op = table.size()
      self.assertAllEqual(2, self.evaluate(size_op))

      save = saver.Saver(var_list=[v0, v1])

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # load table
      for k, v in enumerate(table.tables):
        self.evaluate(
            v.load_from_file_system("file:///tmp/test_local_file_system/" +
                                    str(k),
                                    buffer_size=4096))
      # Check that the parameter nodes have been restored.
      self.assertEqual([10.0], self.evaluate(v0))
      self.assertEqual([20.0], self.evaluate(v1))

      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([5, 0, 1, 2, 6], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([[-1.0], [0.0], [1.0], [2.0], [-1.0]],
                          self.evaluate(output))

      del table

  def test_table_save_load_local_file_system_for_estimator(self):

    def input_fn():
      return {"x": constant_op.constant([1], dtype=dtypes.int64)}

    def model_fn(features, labels, mode, params):
      file_system_saver = de.FileSystemSaver()
      embedding = de.get_variable(
          name="embedding",
          dim=3,
          trainable=False,
          key_dtype=dtypes.int64,
          value_dtype=dtypes.float32,
          initializer=-1.0,
          kv_creator=de.CuckooHashTableCreator(saver=file_system_saver),
      )
      lookup = de.embedding_lookup(embedding, features["x"])
      upsert = embedding.upsert(features["x"],
                                constant_op.constant([[1.0, 2.0, 3.0]]))

      with ops.control_dependencies([lookup, upsert]):
        train_op = state_ops.assign_add(training.get_global_step(), 1)

      scaffold = training.Scaffold(
          saver=saver.Saver(sharded=True,
                            max_to_keep=1,
                            keep_checkpoint_every_n_hours=None,
                            defer_build=True,
                            save_relative_paths=True))
      est = estimator_lib.EstimatorSpec(mode=mode,
                                        scaffold=scaffold,
                                        loss=constant_op.constant(0.),
                                        train_op=train_op,
                                        predictions=lookup)
      return est

    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    # train and save
    try:  # only test for tensorflow <= 2.15
      est = estimator.Estimator(model_fn=model_fn, model_dir=save_path)
      est.train(input_fn=input_fn, steps=1)

      # restore and predict
      predict_results = next(est.predict(input_fn=input_fn))
      self.assertAllEqual(predict_results, [1.0, 2.0, 3.0])
    except:
      pass

  def test_save_restore_only_table(self):
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

      save = saver.Saver(table.tables)
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
      del table

  def test_training_save_restore(self):
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
      id += 1
      save_dir = os.path.join(self.get_temp_dir(), "save_restore")
      save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

      ids = script_ops.py_func(_create_dynamic_shape_tensor(),
                               inp=[],
                               Tout=key_dtype,
                               stateful=True)

      params = de.get_variable(
          name="params-test-0915-" + str(id),
          key_dtype=key_dtype,
          value_dtype=value_dtype,
          initializer=init_ops.random_normal_initializer(0.0, 0.01),
          dim=dim,
      )
      _, var0 = de.embedding_lookup(params,
                                    ids,
                                    return_trainable=True,
                                    name='ff9800')

      def loss():
        return var0 * var0

      params_keys, params_vals = params.export()
      mini = opt.minimize(loss, var_list=[var0])
      opt_slots = params.get_slot_variables(opt)
      saveable_tables = []
      saveable_tables.extend(params.tables)
      for _s in opt_slots:
        saveable_tables.extend(_s.tables)
      _saver = saver.Saver(saveable_tables)

      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        self.evaluate(variables.global_variables_initializer())
        for _i in range(step):
          self.evaluate([mini])
        size_before_saved = self.evaluate(params.size())
        np_params_keys_before_saved = self.evaluate(params_keys)
        np_params_vals_before_saved = self.evaluate(params_vals)
        opt_slots_kv_pairs = [_s.export() for _s in opt_slots]
        np_slots_kv_pairs_before_saved = [
            self.evaluate(_kv) for _kv in opt_slots_kv_pairs
        ]
        _saver.save(sess, save_path)

      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        self.evaluate(variables.global_variables_initializer())
        self.assertAllEqual(0, self.evaluate(params.size()))

        _saver.restore(sess, save_path)
        params_keys_restored, params_vals_restored = params.export()
        size_after_restored = self.evaluate(params.size())
        np_params_keys_after_restored = self.evaluate(params_keys_restored)
        np_params_vals_after_restored = self.evaluate(params_vals_restored)

        opt_slots_kv_pairs_restored = [_s.export() for _s in opt_slots]
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

  def test_import_meta_graph_from_checkpoint(self):
    ops.disable_eager_execution()
    keys_dtype_list = [dtypes.int64]
    values_dtype_list = [dtypes.float32]

    run_id = 0
    for num_shards, k_dtype, d_dtype, init_mode, dim, run_step in itertools.product(
        [2],
        keys_dtype_list,
        values_dtype_list,
        ['constant'],
        [1, 10],
        [10],
    ):
      run_id += 1
      with ops.Graph().as_default() as g:
        with self.session(graph=g,
                          config=default_config,
                          use_gpu=test_util.is_gpu_available()) as sess:
          test_graph = TestGraph(k_dtype, d_dtype, dim, num_shards, 'var',
                                 'dvar', run_id)
          self.evaluate(variables.global_variables_initializer())
          sess.run([test_graph.devar_init_op])
          for _ in range(run_step):
            sess.run([test_graph.var_opt_op, test_graph.devar_opt_op])
          var_loss_name = test_graph.var_loss.name
          devar_loss_name = test_graph.devar_loss.name
          prev_var_loss, prev_devar_loss = sess.run(
              [var_loss_name, devar_loss_name])
          check_de_keys_op_name = test_graph.check_de_keys.name
          prev_keys = sorted(sess.run(check_de_keys_op_name))

          self.assertAllCloseAccordingToType(
              prev_var_loss,
              prev_devar_loss,
              msg='Cond:{},{},{},{},{},{}'.format(num_shards, k_dtype, d_dtype,
                                                  init_mode, dim, run_step))
          _write_checkpoint(self, sess)
      with ops.Graph().as_default() as g:
        with self.session(graph=g,
                          config=default_config,
                          use_gpu=test_util.is_gpu_available()) as sess:
          ckpt_dir = self.get_temp_dir()
          save = saver.import_meta_graph(_get_meta_file(ckpt_dir))
          save.restore(sess, saver.latest_checkpoint(ckpt_dir))

          keys = sorted(sess.run(check_de_keys_op_name))
          var_loss, devar_loss = sess.run([var_loss_name, devar_loss_name])

      self.assertAllEqual(prev_keys, keys)
      self.assertAllCloseAccordingToType(var_loss,
                                         prev_var_loss,
                                         msg='Cond:{},{},{},{},{},{}'.format(
                                             num_shards, k_dtype, d_dtype,
                                             init_mode, dim, run_step))
      self.assertAllCloseAccordingToType(devar_loss,
                                         prev_devar_loss,
                                         msg='Cond:{},{},{},{},{},{}'.format(
                                             num_shards, k_dtype, d_dtype,
                                             init_mode, dim, run_step))

  def test_fail_to_write_checkpoint_for_loaded_meta_graph(self):
    run_id = 0
    for num_shards, k_dtype, d_dtype, init_mode, dim, run_step in itertools.product(
        [2],
        [dtypes.int64],
        [dtypes.float32],
        ['constant'],
        [1, 10],
        [10],
    ):
      run_id += 1
    with ops.Graph().as_default() as g:
      with self.session(graph=g,
                        use_gpu=test_util.is_gpu_available(),
                        config=default_config) as sess:
        graph = TestGraph(k_dtype, d_dtype, dim, num_shards, 'var', 'devar',
                          run_id)
        self.evaluate(variables.global_variables_initializer())
        sess.run([graph.devar_init_op])
        sess.run([graph.var_opt_op, graph.devar_opt_op])
        _write_checkpoint(self, sess)

    with ops.Graph().as_default() as g:
      with self.session(graph=g,
                        use_gpu=test_util.is_gpu_available(),
                        config=default_config) as sess:
        ckpt_dir = self.get_temp_dir()
        save = saver.import_meta_graph(_get_meta_file(ckpt_dir))
        save.restore(sess, saver.latest_checkpoint(ckpt_dir))
        with self.assertRaises(TypeError) as te:
          _write_checkpoint(self, sess)
          self.assertStartsWith(te.exception, "Can't convert Operation")

  def test_object_oriented_checkpoint(self):
    if not context.executing_eagerly():
      self.skipTest('Skip legacy mode for object oriented checkpoint.')

    class TestModel(module.Module):

      def __init__(self):
        self.devar_name = '_devar'
        self.devar = de.get_variable(self.devar_name,
                                     dim=2,
                                     initializer=kinit.RandomNormal(-1, 1))
        self.trainables = data_structures.NoDependency([])
        self.optmz = de.DynamicEmbeddingOptimizer(optimizer_v2.adam.Adam(0.1))
        self.slot_vars = []

      def __call__(self, x):
        embed, tw = de.embedding_lookup(self.devar,
                                        x,
                                        return_trainable=True,
                                        name='ojo2021')
        if not self.trainables:
          self.trainables.append(tw)
        loss = math_ops.reduce_sum(embed)
        return loss

      def train(self, x):

        def _loss_fn():
          return self(x)

        self.optmz.minimize(_loss_fn, var_list=self.trainables)
        if not self.slot_vars:
          self.slot_vars += self.devar.get_slot_variables(self.optmz)

    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      model1 = TestModel()
      features = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
      for _ in range(5):
        model1.train(features)
      loss1 = model1(features)
      keys1, vals1 = model1.devar.export()
      keys1, vals1 = zip(
          *sorted(zip(keys1, vals1), key=lambda x: x[0], reverse=False))
      slot_keys_and_vals1 = [sv.export() for sv in model1.slot_vars]

      ckpt1 = Checkpoint(model=model1, optimizer=model1.optmz)
      ckpt_dir = self.get_temp_dir()
      model_path = ckpt1.save(ckpt_dir)
      del model1

      model2 = TestModel()
      ckpt2 = Checkpoint(model=model2, optimizer=model2.optmz)
      model2.train(features)  # Pre-build trace before restore.
      ckpt2.restore(model_path)
      loss2 = model2(features)
      keys2, vals2 = model2.devar.export()
      keys2, vals2 = zip(
          *sorted(zip(keys2, vals2), key=lambda x: x[0], reverse=False))
      slot_keys_and_vals2 = [sv.export() for sv in model2.slot_vars]
      del model2

      self.assertAllEqual(loss1, loss2)
      self.assertAllEqual(keys1, keys2)
      self.assertAllEqual(vals1, vals2)

      for i in range(len(slot_keys_and_vals1)):
        ont_slot_kvs1 = list(
            zip(*sorted(zip(slot_keys_and_vals1[i][0], slot_keys_and_vals1[i]
                            [1]),
                        key=lambda x: x[0],
                        reverse=False)))
        ont_slot_kvs2 = list(
            zip(*sorted(zip(slot_keys_and_vals2[i][0], slot_keys_and_vals2[i]
                            [1]),
                        key=lambda x: x[0],
                        reverse=False)))
        self.assertAllEqual(ont_slot_kvs1[0], ont_slot_kvs2[0])
        self.assertAllEqual(ont_slot_kvs1[1], ont_slot_kvs2[1])

  def test_get_variable(self):
    with self.session(
        config=default_config,
        graph=ops.Graph(),
        use_gpu=test_util.is_gpu_available(),
    ):
      default_val = -1
      with variable_scope.variable_scope("embedding", reuse=True):
        table1 = de.get_variable("t1",
                                 dtypes.int64,
                                 dtypes.int32,
                                 initializer=default_val,
                                 dim=2)
        table2 = de.get_variable("t1",
                                 dtypes.int64,
                                 dtypes.int32,
                                 initializer=default_val,
                                 dim=2)
        table3 = de.get_variable("t2",
                                 dtypes.int64,
                                 dtypes.int32,
                                 initializer=default_val,
                                 dim=2)

      self.assertAllEqual(table1, table2)
      self.assertNotEqual(table1, table3)

  def test_get_variable_reuse_error(self):
    ops.disable_eager_execution()
    with self.session(
        config=default_config,
        graph=ops.Graph(),
        use_gpu=test_util.is_gpu_available(),
    ):
      with variable_scope.variable_scope("embedding", reuse=False):
        _ = de.get_variable("t900", initializer=-1, dim=2)
        with self.assertRaisesRegexp(ValueError,
                                     "Variable embedding/t900 already exists"):
          _ = de.get_variable("t900", initializer=-1, dim=2)

  @test_util.run_v1_only("Multiple sessions")
  def test_sharing_between_multi_sessions(self):
    ops.disable_eager_execution()
    # Start a server to store the table state
    server = server_lib.Server({"local0": ["localhost:0"]},
                               protocol="grpc",
                               start=True)
    # Create two sessions sharing the same state
    session1 = session.Session(server.target, config=default_config)
    session2 = session.Session(server.target, config=default_config)

    table = de.get_variable("tx100",
                            dtypes.int64,
                            dtypes.int32,
                            initializer=0,
                            dim=1)

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
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()):
      default_val = constant_op.constant([-1, -2], dtypes.int64)
      keys = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      values = constant_op.constant([[0, 1], [2, 3], [4, 5], [6, 7]],
                                    dtypes.int32)
      table = de.get_variable("t10",
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val,
                              dim=2)
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

      del table

  def test_dynamic_embedding_variable_export_insert(self):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()):
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int32)
      table1 = de.get_variable("t101",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val,
                               dim=2)
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
      table2 = de.get_variable("t102",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val,
                               dim=2)
      self.assertAllEqual(0, self.evaluate(table2.size()))
      self.evaluate(table2.upsert(exported_keys, exported_values))
      self.assertAllEqual(3, self.evaluate(table2.size()))

      # Verify lookup result is still the same
      output2 = table2.lookup(input_keys)
      self.assertAllEqual(expected_output, self.evaluate(output2))

  def test_dynamic_embedding_variable_invalid_shape(self):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()):
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      table = de.get_variable("t110",
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val,
                              dim=2)

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

  def test_dynamic_embedding_variable_find_high_rank(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)
      table = de.get_variable("t140",
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_keys = constant_op.constant([[0, 1], [2, 4]], dtypes.int64)
      output = table.lookup(input_keys)
      self.assertAllEqual([2, 2, 1], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual([[[0], [1]], [[2], [-1]]], result)

  def test_dynamic_embedding_variable_insert_low_rank(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
      values = constant_op.constant([[[0], [1]], [[2], [3]]], dtypes.int32)
      table = de.get_variable("t150",
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 3, 4], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertAllEqual([[0], [1], [3], [-1]], result)

  def test_dynamic_embedding_variable_remove_low_rank(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
      values = constant_op.constant([[[0], [1]], [[2], [3]]], dtypes.int32)
      table = de.get_variable("t160",
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val)

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
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = constant_op.constant([-1, -1, -1], dtypes.int32)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                                    dtypes.int32)
      table = de.get_variable("t170",
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val,
                              dim=3)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([[0, 1], [3, 4]], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([2, 2, 3], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual(
          [[[0, 1, 2], [2, 3, 4]], [[-1, -1, -1], [-1, -1, -1]]], result)

  def test_dynamic_embedding_variable_remove_high_rank(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = constant_op.constant([-1, -1, -1], dtypes.int32)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                                    dtypes.int32)
      table = de.get_variable("t180",
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val,
                              dim=3)

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
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)

      table1 = de.get_variable("t191",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val)
      table2 = de.get_variable("t192",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val)
      table3 = de.get_variable("t193",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val)
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
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = constant_op.constant(-1, dtypes.int32)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)
      table = de.get_variable("t200",
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 3], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertAllEqual([[0], [1], [-1]], result)

  def test_signature_mismatch(self):
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    with self.session(config=config, use_gpu=test_util.is_gpu_available()):
      default_val = -1
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)
      table = de.get_variable("t210",
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val)

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
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()):
      default_val = -1.0
      keys = constant_op.constant([3, 7, 0], dtypes.int64)
      values = constant_op.constant([[7.5], [-1.2], [9.9]], dtypes.float32)
      table = de.get_variable("t220",
                              dtypes.int64,
                              dtypes.float32,
                              initializer=default_val)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([7, 0, 11], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertAllClose([[-1.2], [9.9], [default_val]], result)

  def test_dynamic_embedding_variable_with_random_init(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0.0], [1.0], [2.0]], dtypes.float32)
      default_val = init_ops.random_uniform_initializer()
      table = de.get_variable("t230",
                              dtypes.int64,
                              dtypes.float32,
                              initializer=default_val)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 3], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertNotEqual([-1.0], result[2])

  def test_dynamic_embedding_variable_with_restrict_v1(self):
    if context.executing_eagerly():
      self.skipTest('skip eager test when using legacy optimizers.')

    optmz = de.DynamicEmbeddingOptimizer(adam.AdamOptimizer(0.1))
    data_len = 32
    maxval = 256
    num_reserved = 100
    trigger = 150
    embed_dim = 8

    var_guard_by_tstp = de.get_variable(
        'tstp_guard',
        key_dtype=dtypes.int64,
        value_dtype=dtypes.float32,
        initializer=-1.,
        dim=embed_dim,
        init_size=256,
        restrict_policy=de.TimestampRestrictPolicy)

    var_guard_by_freq = de.get_variable(
        'freq_guard',
        key_dtype=dtypes.int64,
        value_dtype=dtypes.float32,
        initializer=-1.,
        dim=embed_dim,
        init_size=256,
        restrict_policy=de.FrequencyRestrictPolicy)

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
    for spv in sparse_vars:
      slot_params += spv.get_slot_variables(optmz)
    slot_params = list(set(slot_params))

    for sp in slot_params:
      self.assertAllEqual(self.evaluate(sp.size()), num_reserved)
    tstp_size = self.evaluate(var_guard_by_tstp.restrict_policy.status.size())
    self.assertAllEqual(tstp_size, num_reserved)
    freq_size = self.evaluate(var_guard_by_freq.restrict_policy.status.size())
    self.assertAllEqual(freq_size, num_reserved)

  def test_dynamic_embedding_variable_with_restrict_v2(self):
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
        'tstp_guard',
        key_dtype=dtypes.int64,
        value_dtype=dtypes.float32,
        initializer=-1.,
        dim=embed_dim,
        restrict_policy=de.TimestampRestrictPolicy)

    var_guard_by_freq = de.get_variable(
        'freq_guard',
        key_dtype=dtypes.int64,
        value_dtype=dtypes.float32,
        initializer=-1.,
        dim=embed_dim,
        restrict_policy=de.FrequencyRestrictPolicy)

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
    for spv in sparse_vars:
      slot_params += spv.get_slot_variables(optmz)
    slot_params = list(set(slot_params))

    for sp in slot_params:
      self.assertAllEqual(sp.size(), num_reserved)
    self.assertAllEqual(var_guard_by_tstp.restrict_policy.status.size(),
                        num_reserved)
    self.assertAllEqual(var_guard_by_freq.restrict_policy.status.size(),
                        num_reserved)

  def test_get_slot_variables(self):
    var = de.get_variable('trl4397',
                          key_dtype=dtypes.int64,
                          value_dtype=dtypes.float32,
                          restrict_policy=de.TimestampRestrictPolicy)
    ids = constant_op.constant([1, 2, 3], dtype=dtypes.int64)

    def loss_fn(name, tws=[]):
      embed, tw = de.embedding_lookup(var,
                                      ids,
                                      return_trainable=True,
                                      name=name)
      tws.clear()
      tws.append(tw)
      loss = math_ops.reduce_sum(embed)
      return loss

    opt1 = de.DynamicEmbeddingOptimizer(adam.AdamOptimizer(0.1))
    opt2 = de.DynamicEmbeddingOptimizer(optimizer_v2.adam.Adam(0.1))

    tws = []
    if context.executing_eagerly():
      opt1.minimize(lambda: loss_fn('g1', tws), var_list=tws)
      opt2.minimize(lambda: loss_fn('g2', tws), var_list=tws)
    else:
      opt1.minimize(loss_fn('g1', tws), var_list=tws)
      opt2.minimize(lambda: loss_fn('g2', tws), var_list=tws)

    result = [v.name for v in var.get_slot_variables(opt1)]
    expect = sorted([v.params.name for v in list(opt1._slots['m'].values())] +
                    [v.params.name for v in list(opt1._slots['v'].values())])
    self.assertAllEqual(result, expect)

    result = [v.name for v in var.get_slot_variables(opt2)]
    expect = sorted([v['m'].params.name for v in list(opt2._slots.values())] +
                    [v['v'].params.name for v in list(opt2._slots.values())])
    self.assertAllEqual(result, expect)


if __name__ == "__main__":
  test.main()
