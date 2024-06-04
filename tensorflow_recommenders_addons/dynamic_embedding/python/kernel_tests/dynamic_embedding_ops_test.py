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
"""unit tests of embedding_lookup APIs
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import itertools
import math
import numpy as np
import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow.python.ops.ragged import ragged_tensor

from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.ragged_embedding_ops import embedding_lookup_sparse, \
  safe_embedding_lookup_sparse

try:
  from tensorflow.python.keras.initializers import initializers_v2 as kinit2
except ImportError:
  kinit2 = None
  pass  # for compatible with TF < 2.3.x

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.keras import initializers as keras_init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
try:  # tf version >= 2.14.0
  from tensorflow.python.ops.array_ops_stack import stack
except:
  from tensorflow.python.ops.array_ops import stack
from tensorflow.python.platform import test
from tensorflow.python.training import device_setter
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat
try:
  from tensorflow.keras.legacy.optimizers import Adam
except:
  from tensorflow.keras.optimizers import Adam


# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
def _type_converter(tf_type):
  mapper = {
      dtypes.int32: np.int32,
      dtypes.int64: np.int64,
      dtypes.float32: float,
      dtypes.float64: np.float64,
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


def _ids_and_weights_2d(embed_dim=4, ragged=False):
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
  if ragged:
    sparse_ids = ragged_tensor.RaggedTensor.from_sparse(sparse_ids)
    sparse_weights = ragged_tensor.RaggedTensor.from_sparse(sparse_weights)
  return sparse_ids, sparse_weights


def _ids_and_weights_3d(
    embed_dim=4) -> (sparse_tensor.SparseTensor, sparse_tensor.SparseTensor):
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
    allow_soft_placement=True,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))


@test_util.deprecated_graph_mode_only
class EmbeddingLookupTest(test.TestCase):

  def test_simple_sharded(self):
    embeddings = de.get_variable(
        "t300",
        dtypes.int64,
        dtypes.float32,
        devices=_get_devices() * 2,
        initializer=2.0,
    )

    ids = constant_op.constant([0, 1, 2, 3, 4], dtype=dtypes.int64)
    embedding, trainable = de.embedding_lookup(embeddings,
                                               ids,
                                               max_norm=1.0,
                                               return_trainable=True)
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      self.assertAllClose(
          embedding.eval(),
          [
              [1.0],
          ] * 5,
      )
      self.evaluate(trainable.update_op())
      self.assertAllEqual(embeddings.size().eval(), 5)
      self.assertAllEqual(embeddings.size(0).eval(), 3)
      self.assertAllEqual(embeddings.size(1).eval(), 2)

  def test_max_norm(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      embeddings = de.get_variable("t310",
                                   dtypes.int64,
                                   dtypes.float32,
                                   initializer=2.0)

      ids = constant_op.constant([0], dtype=dtypes.int64)
      embedding = de.embedding_lookup(embeddings, ids, max_norm=1.0)
      self.assertAllEqual(embedding.eval(), [[1.0]])

  def test_max_norm_nontrivial(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      embeddings = de.get_variable("t320",
                                   dtypes.int64,
                                   dtypes.float32,
                                   initializer=2.0,
                                   dim=2)
      fake_values = constant_op.constant([[2.0, 4.0], [3.0, 1.0]])
      ids = constant_op.constant([0, 1], dtype=dtypes.int64)
      self.evaluate(embeddings.upsert(ids, fake_values))
      embedding_no_norm = de.embedding_lookup(embeddings, ids)
      embedding = de.embedding_lookup(embeddings, ids, max_norm=2.0)
      norms = math_ops.sqrt(
          math_ops.reduce_sum(embedding_no_norm * embedding_no_norm, axis=1))
      normalized = embedding_no_norm / stack([norms, norms], axis=1)
      self.assertAllCloseAccordingToType(embedding.eval(),
                                         2 * self.evaluate(normalized))

  def test_sharded_custom_partitioner_int32_ids(self):

    def _partition_fn(keys, shard_num):
      return math_ops.cast(keys % 2, dtype=dtypes.int32)

    embeddings = de.get_variable(
        "t330",
        dtypes.int64,
        dtypes.float32,
        partitioner=_partition_fn,
        devices=_get_devices() * 3,
        initializer=2.0,
    )

    ids = constant_op.constant([0, 1, 2, 3, 4], dtype=dtypes.int64)
    vals = constant_op.constant([[0.0], [1.0], [2.0], [3.0], [4.0]],
                                dtype=dtypes.float32)
    ids_test = constant_op.constant([1, 3, 2, 3, 0], dtype=dtypes.int64)
    embedding = de.embedding_lookup(embeddings, ids_test)
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      self.evaluate(embeddings.upsert(ids, vals))
      self.assertAllClose(embedding.eval(), [[1.0], [3.0], [2.0], [3.0], [0.0]])
      self.assertAllEqual([5, 1], embedding.eval().shape)
      self.assertAllEqual(3, embeddings.size(0).eval())
      self.assertAllEqual(2, embeddings.size(1).eval())
      self.assertAllEqual(0, embeddings.size(2).eval())

  def test_sharded_multi_lookup_on_one_variable(self):
    embeddings = de.get_variable(
        "t340",
        dtypes.int64,
        dtypes.float32,
        devices=_get_devices() * 3,
        initializer=2.0,
    )

    ids = constant_op.constant([0, 1, 2, 3, 4], dtype=dtypes.int64)
    vals = constant_op.constant([[0.0], [1.0], [2.0], [3.0], [4.0]],
                                dtype=dtypes.float32)
    new_vals = constant_op.constant([[10.0], [11.0], [12.0], [13.0], [14.0]],
                                    dtype=dtypes.float32)

    ids0 = constant_op.constant([1, 3, 2], dtype=dtypes.int64)
    ids1 = constant_op.constant([3, 4], dtype=dtypes.int64)

    embedding0 = de.embedding_lookup(embeddings, ids0)
    embedding1 = de.embedding_lookup(embeddings, ids1)

    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      self.evaluate(embeddings.upsert(ids, vals))
      self.assertAllClose(embedding0.eval(), [[1.0], [3.0], [2.0]])
      self.assertAllEqual([3, 1], embedding0.eval().shape)
      self.assertAllClose(embedding1.eval(), [[3.0], [4.0]])
      self.assertAllEqual([2, 1], embedding1.eval().shape)
      self.evaluate(embeddings.upsert(ids, new_vals))
      self.assertAllClose(embedding1.eval(), [[13.0], [14.0]])
      self.assertAllEqual([2, 1], embedding1.eval().shape)

  def test_higher_rank(self):
    np.random.seed(8)
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      for dim in [1, 10]:
        for ids_shape in [[3, 2], [4, 3], [4, 3, 10]]:
          with variable_scope.variable_scope("test_higher_rank", reuse=True):
            params = de.get_variable(
                "t350-" + str(dim),
                dtypes.int64,
                dtypes.float32,
                initializer=2.0,
                dim=dim,
            )
            ids = np.random.randint(2**31, size=np.prod(ids_shape),
                                    dtype=int).reshape(ids_shape)
            ids = constant_op.constant(ids, dtype=dtypes.int64)
            simple = params.lookup(ids)
            self.evaluate(params.upsert(ids, simple))

            embedding = de.embedding_lookup(params, ids)
            self.assertAllEqual(simple.eval(), embedding.eval())
            self.assertAllEqual(ids_shape + [dim], embedding.eval().shape)

  def test_static_shape_checking(self):
    np.random.seed(8)
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      for dim in [1, 10]:
        for ids_shape in [[3, 2], [4, 3], [4, 3, 10]]:
          with variable_scope.variable_scope(
              "test_static_shape_checking" + str(dim),
              reuse=variable_scope.AUTO_REUSE,
          ):
            params = de.get_variable(
                "test_static_shape_checking-" + str(dim),
                dtypes.int64,
                dtypes.float32,
                initializer=2.0,
                dim=dim,
            )
            params_nn = variable_scope.get_variable("n",
                                                    shape=[100, dim],
                                                    use_resource=False)
            ids = np.random.randint(2**31, size=np.prod(ids_shape),
                                    dtype=int).reshape(ids_shape)
            ids = constant_op.constant(ids, dtype=dtypes.int64)

            embedding_test = de.embedding_lookup(params, ids)
            embedding_base = embedding_ops.embedding_lookup(params_nn, ids)
            self.assertAllEqual(embedding_test.shape, embedding_base.shape)

  def test_dynamic_shape_checking(self):
    np.random.seed(8)
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      for dim in [1, 10]:
        for ids_shape in [None, [-1, 1], [1, -1, 1], [-1, 1, 1]]:
          with variable_scope.variable_scope(
              "test_static_shape_checking" + str(dim),
              reuse=variable_scope.AUTO_REUSE,
          ):
            params = de.get_variable(
                "test_static_shape_checking-" + str(dim),
                dtypes.int64,
                dtypes.float32,
                initializer=2.0,
                dim=dim,
            )
            params_nn = variable_scope.get_variable("n",
                                                    shape=[100, dim],
                                                    use_resource=False)
            ids = script_ops.py_func_common(
                _create_dynamic_shape_tensor(min_val=0, max_val=100),
                inp=[],
                Tout=dtypes.int64,
                stateful=True,
            )
            if ids_shape is not None:
              ids = array_ops.reshape(ids, ids_shape)

            embedding_test = de.embedding_lookup(params, ids)
            embedding_base = embedding_ops.embedding_lookup(params_nn, ids)

            # check static shape
            if ids_shape is None:
              # ids with unknown shape
              self.assertTrue(embedding_test.shape == embedding_base.shape)
            else:
              # ids with no fully-defined shape.
              self.assertAllEqual(
                  embedding_test.shape.as_list(),
                  embedding_base.shape.as_list(),
              )

            self.evaluate(variables.global_variables_initializer())

            # check static shape
            for _ in range(10):
              embedding_test_val, embedding_base_val = self.evaluate(
                  [embedding_test, embedding_base])
              self.assertAllEqual(embedding_test_val.shape,
                                  embedding_base_val.shape)

  def test_scope_reuse_embedding_lookup(self):
    ids = constant_op.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               dtype=dtypes.int64)
    with variable_scope.variable_scope("test", reuse=variable_scope.AUTO_REUSE):
      p1 = de.get_variable(name="p1")
      with variable_scope.variable_scope("q"):
        _, t1 = de.embedding_lookup(p1, ids, name="emb", return_trainable=True)

    with variable_scope.variable_scope("test", reuse=variable_scope.AUTO_REUSE):
      p1_reuse = de.get_variable(name="p1")
      p2 = de.get_variable(name="p2")
      with variable_scope.variable_scope("q"):
        _, t2 = de.embedding_lookup(p2, ids, name="emb", return_trainable=True)

    self.assertAllEqual(p1.name, "test/p1")
    self.assertAllEqual(p2.name, "test/p2")
    self.assertAllEqual(p1, p1_reuse)
    self.assertEqual(t1.name, "test/q/emb/emb:0")
    self.assertEqual(t2.name, "test/q/emb/emb_1:0")
    self.assertAllEqual(p1._tables[0].name, "test_p1_mht_1of1")
    self.assertAllEqual(p1_reuse._tables[0].name, "test_p1_mht_1of1")
    self.assertAllEqual(p2._tables[0].name, "test_p2_mht_1of1")

  def test_scope_reuse_sparse_embedding_lookup(self):
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
    shape = [2, 3, 4]

    sparse_ids = sparse_tensor.SparseTensor(
        constant_op.constant(indices, dtypes.int64),
        constant_op.constant(ids, dtypes.int64),
        constant_op.constant(shape, dtypes.int64),
    )

    with variable_scope.variable_scope("test", reuse=variable_scope.AUTO_REUSE):
      p1 = de.get_variable(name="p1")
      with variable_scope.variable_scope("q"):
        _, t1 = de.embedding_lookup_sparse(p1,
                                           sparse_ids,
                                           None,
                                           name="sp_emb",
                                           return_trainable=True)

    with variable_scope.variable_scope("test", reuse=variable_scope.AUTO_REUSE):
      p1_reuse = de.get_variable(name="p1")
      p2 = de.get_variable(name="p2")
      with variable_scope.variable_scope("q"):
        _, t2 = de.embedding_lookup_sparse(p2,
                                           sparse_ids,
                                           None,
                                           name="sp_emb",
                                           return_trainable=True)

    self.assertAllEqual(p1.name, "test/p1")
    self.assertAllEqual(p2.name, "test/p2")
    self.assertAllEqual(p1, p1_reuse)
    self.assertEqual(
        t1.name, "test/q/sp_emb/embedding_lookup/sp_emb/embedding_lookup:0")
    self.assertEqual(
        t2.name, "test/q/sp_emb/embedding_lookup/sp_emb/embedding_lookup_1:0")
    self.assertAllEqual(p1._tables[0].name, "test_p1_mht_1of1")
    self.assertAllEqual(p1_reuse._tables[0].name, "test_p1_mht_1of1")
    self.assertAllEqual(p2._tables[0].name, "test_p2_mht_1of1")

  def test_scope_reuse_safe_sparse_embedding_lookup(self):
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
    shape = [2, 3, 4]

    sparse_ids = sparse_tensor.SparseTensor(
        constant_op.constant(indices, dtypes.int64),
        constant_op.constant(ids, dtypes.int64),
        constant_op.constant(shape, dtypes.int64),
    )

    with variable_scope.variable_scope("test", reuse=variable_scope.AUTO_REUSE):
      p1 = de.get_variable(name="p1")
      with variable_scope.variable_scope("q"):
        _, t1 = de.safe_embedding_lookup_sparse(p1,
                                                sparse_ids,
                                                None,
                                                name="safe_sp_emb",
                                                return_trainable=True)

    with variable_scope.variable_scope("test", reuse=variable_scope.AUTO_REUSE):
      p1_reuse = de.get_variable(name="p1")
      p2 = de.get_variable(name="p2")
      with variable_scope.variable_scope("q"):
        _, t2 = de.safe_embedding_lookup_sparse(p2,
                                                sparse_ids,
                                                None,
                                                name="safe_sp_emb",
                                                return_trainable=True)

    self.assertAllEqual(p1.name, "test/p1")
    self.assertAllEqual(p2.name, "test/p2")
    self.assertAllEqual(p1, p1_reuse)
    self.assertEqual(
        t1.name,
        "test/q/safe_sp_emb/embedding_lookup_sparse/embedding_lookup/safe_sp_emb/embedding_lookup_sparse/embedding_lookup:0",
    )
    self.assertEqual(
        t2.name,
        "test/q/safe_sp_emb/embedding_lookup_sparse/embedding_lookup/safe_sp_emb/embedding_lookup_sparse/embedding_lookup_1:0",
    )
    self.assertAllEqual(p1._tables[0].name, "test_p1_mht_1of1")
    self.assertAllEqual(p1_reuse._tables[0].name, "test_p1_mht_1of1")
    self.assertAllEqual(p2._tables[0].name, "test_p2_mht_1of1")

  def test_treated_as_worker_op_by_device_setter(self):
    num_ps_tasks = 2
    with ops.device("/job:worker/task:0"):
      ids = constant_op.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                 dtype=dtypes.int64)
    setter = device_setter.replica_device_setter(ps_tasks=num_ps_tasks,
                                                 ps_device="/job:ps",
                                                 worker_device="/job:worker")
    with ops.device(setter):
      p1 = de.get_variable(name="p1",
                           devices=["/job:ps/task:0", "/job:ps/task:1"])
      _ = de.embedding_lookup(p1, ids, name="emb")
    self.assertTrue("/job:ps/task:0" in p1._tables[0].resource_handle.device)
    self.assertTrue("/job:ps/task:1" in p1._tables[1].resource_handle.device)

  def test_embedding_lookup_sparse_with_initializer(self):
    id = 0
    embed_dim = 8
    elements_num = 262144
    test_util.random_seed.set_seed(2021)
    init_list = [
        (init_ops.random_normal_initializer(0.0, 0.001), 0.0, 0.001),
        (init_ops.truncated_normal_initializer(0.0, 0.001), 0.0, 0.00088),
        (keras_init_ops.RandomNormalV2(mean=0.0, stddev=0.001), 0.0, 0.001),
        (keras_init_ops.GlorotNormal(), 0.0, 0.004784),
    ]
    if kinit2 is not None and hasattr(kinit2, "GlorotNormal"):
      init_list.append((kinit2.GlorotNormal(), 0.0, 0.004784))
    for initializer, target_mean, target_stddev in init_list:
      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()):
        id += 1
        embedding_weights = de.get_variable(
            "emb-init-bugfix-" + str(id),
            key_dtype=dtypes.int64,
            value_dtype=dtypes.float32,
            devices=_get_devices() * 3,
            initializer=initializer,
            dim=embed_dim,
        )

        ids = np.random.randint(
            -0x7FFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF,
            elements_num,
            dtype=np.int64,
        )
        ids = np.unique(ids)
        ids = constant_op.constant(ids, dtypes.int64)
        vals_op = de.embedding_lookup(embedding_weights, ids, None).eval()

        mean = self.evaluate(math_ops.reduce_mean(vals_op))
        stddev = self.evaluate(math_ops.reduce_std(vals_op))
        rtol = 2e-5
        atol = rtol
        self.assertTrue(not (list(vals_op[0]) == list(vals_op[1])))
        self.assertAllClose(target_mean, mean, rtol, atol)
        self.assertAllClose(target_stddev, stddev, rtol, atol)

  def test_embedding_lookup_shape(self):

    def _evaluate(tensors, feed_dict):
      sess = ops.get_default_session()
      if sess is None:
        with self.test_session() as sess:
          return sess.run(tensors, feed_dict=feed_dict)
      else:
        return sess.run(tensors, feed_dict=feed_dict)

    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1

      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                                    dtypes.int32)
      table = de.get_variable("t140",
                              dtypes.int64,
                              dtypes.int32,
                              dim=3,
                              initializer=default_val)
      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      # shape of ids is fully defined
      ids = constant_op.constant([[0, 1], [2, 4]], dtypes.int64)
      embeddings = de.embedding_lookup(table, ids)
      self.assertAllEqual([2, 2, 3], embeddings.get_shape())
      re = self.evaluate(embeddings)
      self.assertAllEqual([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [-1, -1, -1]]],
                          re)

      # shape of ids is partially defined
      ids = gen_array_ops.placeholder(shape=(2, None), dtype=dtypes.int64)
      embeddings = de.embedding_lookup(table, ids)
      self.assertFalse(embeddings.get_shape().is_fully_defined())
      re = _evaluate(
          embeddings,
          feed_dict={ids: np.asarray([[0, 1], [2, 4]], dtype=np.int64)})
      self.assertAllEqual([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [-1, -1, -1]]],
                          re)

      # shape of ids is unknown
      ids = gen_array_ops.placeholder(dtype=dtypes.int64)
      embeddings = de.embedding_lookup(table, ids)
      self.assertEqual(embeddings.get_shape(), tensor_shape.unknown_shape())
      re = _evaluate(
          embeddings,
          feed_dict={ids: np.asarray([[0, 1], [2, 4]], dtype=np.int64)})
      self.assertAllEqual([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [-1, -1, -1]]],
                          re)


@test_util.deprecated_graph_mode_only
class EmbeddingLookupUniqueTest(test.TestCase):

  def test_embedding_lookup_unique(self):
    dim = 5
    n = 10
    embeddings_de = de.get_variable("t_unique_001",
                                    dtypes.int64,
                                    dtypes.float32,
                                    dim=dim)
    ids_shape = (2, 3, 4)
    embeddings_np = np.random.randn(n, dim)
    ids = np.random.randint(0, n, ids_shape)

    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      self.evaluate(embeddings_de.upsert(list(range(n)), embeddings_np))
      embedded_np = embeddings_np[ids]
      embedded_de = de.embedding_lookup_unique(embeddings_de, ids).eval()

    self.assertEqual(embedded_np.shape, embedded_de.shape)
    np.testing.assert_almost_equal(embedded_np, embedded_de)


class EmbeddingLookupSparseTest(test.TestCase, parameterized.TestCase):

  def _random_ids_and_weights(self,
                              batch_size,
                              vocab_size,
                              k_type,
                              d_type,
                              ragged=False):
    max_val_per_entry = 6
    vals_per_batch_entry = np.random.randint(1,
                                             max_val_per_entry,
                                             size=batch_size)
    num_vals = np.sum(vals_per_batch_entry)

    ids = np.random.randint(vocab_size, size=num_vals)
    ids = ids.astype(_type_converter(k_type))
    weights = 1 + np.random.rand(num_vals)
    weights = weights.astype(_type_converter(d_type))

    indices = []
    for batch_entry, num_val in enumerate(vals_per_batch_entry):
      for val_index in range(num_val):
        indices.append([batch_entry, val_index])

    shape = [batch_size, max_val_per_entry]

    sp_ids = sparse_tensor.SparseTensor(
        constant_op.constant(indices, dtypes.int64),
        constant_op.constant(ids, k_type),
        constant_op.constant(shape, dtypes.int64),
    )
    sp_weights = sparse_tensor.SparseTensor(
        constant_op.constant(indices, dtypes.int64),
        constant_op.constant(weights, d_type),
        constant_op.constant(shape, dtypes.int64),
    )
    if ragged:
      sp_ids = ragged_tensor.RaggedTensor.from_sparse(sp_ids)
      sp_weights = ragged_tensor.RaggedTensor.from_sparse(sp_weights)
    return sp_ids, sp_weights, ids, weights, vals_per_batch_entry

  def _group_by_batch_entry(self, vals, vals_per_batch_entry):
    grouped_vals = []
    index = 0
    for num_val in vals_per_batch_entry:
      grouped_vals.append(list(vals[index:(index + num_val)]))
      index += num_val
    return grouped_vals

  @test_util.run_all_in_graph_and_eager_modes
  @parameterized.parameters(itertools.product([True, False]))
  def test_embedding_lookup_sparse(self, ragged):
    var_id = 0
    for (
        num_shards,
        initial_mode,
        combiner,
        k_dtype,
        d_dtype,
        ignore_weights,
        dim,
    ) in itertools.product(
        [1, 3],
        ["constant", "random"],
        [
            "sum",
            "mean",
            "sqrtn",
        ],
        [dtypes.int64],
        [
            dtypes.float32,
        ],
        [True, False],
        [1, 5],
    ):
      vocab_size = 2**31 if k_dtype == dtypes.int32 else 2**63
      batch_size = 5

      (
          sp_ids,
          sp_weights,
          ids,
          weights,
          vals_per_batch_entry,
      ) = self._random_ids_and_weights(batch_size, vocab_size, k_dtype, d_dtype,
                                       ragged)

      grouped_ids = self._group_by_batch_entry(ids, vals_per_batch_entry)
      grouped_weights = self._group_by_batch_entry(weights,
                                                   vals_per_batch_entry)
      grouped_ignored_weights = self._group_by_batch_entry(
          np.ones(np.sum(vals_per_batch_entry)), vals_per_batch_entry)

      with self.session(use_gpu=test_util.is_gpu_available(),
                        config=default_config):
        var_id += 1
        params = de.get_variable(
            "t1000-" + str(var_id),
            key_dtype=k_dtype,
            value_dtype=d_dtype,
            devices=_get_devices() * num_shards,
            initializer=1.0,
            dim=dim,
        )
        random_init = params.lookup(ids)
        init_op = params.upsert(ids, random_init)
        self.evaluate(init_op)
        np_params = random_init.numpy() if context.executing_eagerly(
        ) else random_init.eval()
        grouped_params = self._group_by_batch_entry(np_params,
                                                    vals_per_batch_entry)
        if ragged:
          embedding_sum = embedding_lookup_sparse(
              params,
              sp_ids,
              None if ignore_weights else sp_weights,
              combiner=combiner,
          )
        else:
          embedding_sum = de.embedding_lookup_sparse(
              params,
              sp_ids,
              None if ignore_weights else sp_weights,
              combiner=combiner,
          )
        self.assertEqual(embedding_sum.dtype, d_dtype)

        tf_embedding_sum = embedding_sum.numpy() if context.executing_eagerly(
        ) else embedding_sum.eval()

        np_embedding_sum, np_weight_sum, np_weight_sq_sum = embedding_result(
            grouped_params,
            grouped_ids,
            weight_vals=grouped_ignored_weights
            if ignore_weights else grouped_weights,
        )
        if combiner == "mean":
          np_embedding_sum /= np.reshape(np_weight_sum, (batch_size, 1))
        if combiner == "sqrtn":
          np_embedding_sum /= np.reshape(np.sqrt(np_weight_sq_sum),
                                         (batch_size, 1))

        rtol = 1e-6
        atol = rtol
        self.assertAllClose(np_embedding_sum, tf_embedding_sum, rtol, atol)

  @test_util.run_all_in_graph_and_eager_modes
  def test_embedding_lookup_sparse_shape_checking(self):
    if context.executing_eagerly():
      self.skipTest("Skip eager test")
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      embed_dim = 4
      embedding_weights_nn = variable_scope.get_variable("n",
                                                         shape=[100, embed_dim],
                                                         use_resource=False)
      embedding_weights_de = _random_weights(embed_dim=4)
      sparse_ids, _ = _ids_and_weights_3d(embed_dim=embed_dim)

      embedding_lookup_base = embedding_ops.embedding_lookup_sparse(
          embedding_weights_nn, sparse_ids, None)
      embedding_lookup_test = de.embedding_lookup_sparse(
          embedding_weights_de, sparse_ids, None)
      self.assertTrue(embedding_lookup_base.get_shape().as_list() ==
                      embedding_lookup_test.get_shape().as_list())


class SafeEmbeddingLookupSparseTest(test.TestCase, parameterized.TestCase):

  def _get_ids_and_weights_3d(self, valid_ids):
    embedding_weights = _random_weights()
    sparse_ids, sparse_weights = _ids_and_weights_3d()

    # init
    embedding_weights_values = embedding_weights.lookup(valid_ids)
    embedding_weights_values = embedding_weights_values.numpy(
    ) if context.executing_eagerly() else embedding_weights_values.eval()
    self.evaluate(embedding_weights.upsert(valid_ids, embedding_weights_values))
    return embedding_weights, embedding_weights_values, sparse_ids, sparse_weights

  @test_util.run_all_in_graph_and_eager_modes
  @parameterized.parameters(itertools.product([True, False]))
  def test_safe_embedding_lookup_sparse_return_zero_vector(self, ragged=False):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      dim = 4
      embedding_weights = _random_weights(embed_dim=dim)
      sparse_ids, sparse_weights = _ids_and_weights_2d(embed_dim=dim,
                                                       ragged=ragged)
      valid_ids = np.array([
          0,
          1,
          2,
          -1,
      ])

      # init
      weights = embedding_weights.lookup(valid_ids)
      embedding_weights_values = weights.numpy() if context.executing_eagerly(
      ) else weights.eval()
      self.evaluate(
          embedding_weights.upsert(valid_ids, embedding_weights_values))

      if ragged:
        embedding_lookup_result = safe_embedding_lookup_sparse(
            embedding_weights, sparse_ids, sparse_weights)
      else:
        embedding_lookup_result = de.safe_embedding_lookup_sparse(
            embedding_weights, sparse_ids, sparse_weights)
      embedding_lookup_result = embedding_lookup_result.numpy(
      ) if context.executing_eagerly() else embedding_lookup_result.eval()

      self.assertAllClose(
          embedding_lookup_result,
          [
              (1.0 * embedding_weights_values[0] +
               2.0 * embedding_weights_values[1] +
               1.0 * embedding_weights_values[3]) / 4.0,
              embedding_weights_values[3] * 1.0,
              [0] * dim,
              embedding_weights_values[2],
              [0] * dim,
          ],
      )

  @test_util.run_all_in_graph_and_eager_modes
  @parameterized.parameters(itertools.product([True, False]))
  def test_safe_embedding_lookup_sparse_return_special_vector(
      self, ragged=False):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      dim = 4
      embedding_weights = _random_weights(embed_dim=dim)
      sparse_ids, sparse_weights = _ids_and_weights_2d(embed_dim=dim,
                                                       ragged=ragged)
      valid_ids = np.array([0, 1, 2, 3, -1])

      # init
      weights = embedding_weights.lookup(valid_ids)
      embedding_weights_values = weights.numpy() if context.executing_eagerly(
      ) else weights.eval()
      self.evaluate(
          embedding_weights.upsert(valid_ids, embedding_weights_values))
      if ragged:
        embedding_lookup_result = safe_embedding_lookup_sparse(
            embedding_weights, sparse_ids, sparse_weights, default_id=3)
      else:
        embedding_lookup_result = de.safe_embedding_lookup_sparse(
            embedding_weights, sparse_ids, sparse_weights, default_id=3)
      embedding_lookup_result = embedding_lookup_result.numpy(
      ) if context.executing_eagerly() else embedding_lookup_result.eval()
      self.assertAllClose(
          embedding_lookup_result,
          [
              (1.0 * embedding_weights_values[0] +
               2.0 * embedding_weights_values[1] +
               1.0 * embedding_weights_values[4]) / 4.0,
              embedding_weights_values[4],
              embedding_weights_values[3],
              embedding_weights_values[2],
              embedding_weights_values[3],
          ],
      )

  @test_util.run_all_in_graph_and_eager_modes
  @parameterized.parameters(itertools.product([True, False]))
  def test_safe_embedding_lookup_sparse_no_weights(self, ragged=False):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      dim = 4
      embedding_weights = _random_weights(embed_dim=dim)
      sparse_ids, sparse_weights = _ids_and_weights_2d(embed_dim=dim,
                                                       ragged=ragged)
      valid_ids = np.array([0, 1, 2, -1])

      # init
      weights = embedding_weights.lookup(valid_ids)
      embedding_weights_values = weights.numpy() if context.executing_eagerly(
      ) else weights.eval()
      self.evaluate(
          embedding_weights.upsert(valid_ids, embedding_weights_values))
      if ragged:
        embedding_lookup_result = safe_embedding_lookup_sparse(
            embedding_weights, sparse_ids, None)
      else:
        embedding_lookup_result = de.safe_embedding_lookup_sparse(
            embedding_weights, sparse_ids, None)
      embedding_lookup_result = embedding_lookup_result.numpy(
      ) if context.executing_eagerly() else embedding_lookup_result.eval()

      self.assertAllClose(
          embedding_lookup_result,
          [
              (embedding_weights_values[0] + embedding_weights_values[1] +
               embedding_weights_values[3]) / 3.0,
              embedding_weights_values[3],
              [0] * 4,
              embedding_weights_values[2],
              (embedding_weights_values[0] + embedding_weights_values[1]) / 2.0,
          ],
      )

  @test_util.run_all_in_graph_and_eager_modes
  @parameterized.parameters(itertools.product([True, False]))
  def test_safe_embedding_lookup_sparse_partitioned(self, ragged=False):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      dim = 4
      embedding_weights = _random_weights(embed_dim=dim, num_shards=3)
      sparse_ids, sparse_weights = _ids_and_weights_2d(embed_dim=dim,
                                                       ragged=ragged)
      valid_ids = np.array([0, 1, 2, -1])

      # init
      weights = embedding_weights.lookup(valid_ids)
      embedding_weights_values = weights.numpy() if context.executing_eagerly(
      ) else weights.eval()
      self.evaluate(
          embedding_weights.upsert(valid_ids, embedding_weights_values))
      if ragged:
        embedding_lookup_result = safe_embedding_lookup_sparse(
            embedding_weights, sparse_ids, None)
      else:
        embedding_lookup_result = de.safe_embedding_lookup_sparse(
            embedding_weights, sparse_ids, None)
      embedding_lookup_result = embedding_lookup_result.numpy(
      ) if context.executing_eagerly() else embedding_lookup_result.eval()

      self.assertAllClose(
          embedding_lookup_result,
          [
              (embedding_weights_values[0] + embedding_weights_values[1] +
               embedding_weights_values[3]) / 3.0,
              embedding_weights_values[3],
              [0] * 4,
              embedding_weights_values[2],
              (embedding_weights_values[0] + embedding_weights_values[1]) / 2.0,
          ],
      )

  @test_util.run_all_in_graph_and_eager_modes
  @parameterized.parameters(itertools.product([True, False]))
  def test_safe_embedding_lookup_sparse_inconsistent_ids_type(
      self, ragged=False):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):

      def fn():
        embedding_weights = _random_weights(num_shards=3,
                                            key_dtype=dtypes.int32)
        sparse_ids, sparse_weights = _ids_and_weights_2d(ragged=ragged)
        if ragged:
          safe_embedding_lookup_sparse(embedding_weights, sparse_ids,
                                       sparse_weights)
        else:
          de.safe_embedding_lookup_sparse(embedding_weights, sparse_ids,
                                          sparse_weights)

      self.assertRaises(TypeError, fn)

  @test_util.run_all_in_graph_and_eager_modes
  @parameterized.parameters(itertools.product([True, False]))
  def test_safe_embedding_lookup_sparse_inconsistent_weights_type(
      self, ragged=False):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):

      def fn():
        embedding_weights = _random_weights(num_shards=3, key_dtype=dtypes.half)
        sparse_ids, sparse_weights = _ids_and_weights_2d(ragged=ragged)
        if ragged:
          safe_embedding_lookup_sparse(embedding_weights, sparse_ids,
                                       sparse_weights)
        else:
          de.safe_embedding_lookup_sparse(embedding_weights, sparse_ids,
                                          sparse_weights)

      self.assertRaises(TypeError, fn)

  @test_util.run_all_in_graph_and_eager_modes
  def test_safe_embedding_lookup_sparse_3d_return_zero_vector(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      valid_ids = np.array([0, 1, 2, -1])
      embedding_weights, embedding_weights_values, sparse_ids, sparse_weights = self._get_ids_and_weights_3d(
          valid_ids)

      embedding_lookup_result = de.safe_embedding_lookup_sparse(
          embedding_weights, sparse_ids, sparse_weights)
      embedding_lookup_result = embedding_lookup_result.numpy(
      ) if context.executing_eagerly() else embedding_lookup_result.eval()

      self.assertAllClose(
          embedding_lookup_result,
          [
              [
                  (1.0 * embedding_weights_values[0] +
                   2.0 * embedding_weights_values[1] +
                   1.0 * embedding_weights_values[3]) / 4.0,
                  embedding_weights_values[3],
                  [0] * 4,
              ],
              [embedding_weights_values[2], [0] * 4, [0] * 4],
          ],
      )

  @test_util.run_all_in_graph_and_eager_modes
  def test_safe_embedding_lookup_sparse_3d_return_special_vector(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      embedding_weights, embedding_weights_values, sparse_ids, sparse_weights = self._get_ids_and_weights_3d(
          np.array([0, 1, 2, 3, -1]))
      embedding_lookup_result = de.safe_embedding_lookup_sparse(
          embedding_weights, sparse_ids, sparse_weights, default_id=3)
      embedding_lookup_result = embedding_lookup_result.numpy(
      ) if context.executing_eagerly() else embedding_lookup_result.eval()
      self.assertAllClose(
          embedding_lookup_result,
          [
              [
                  (1.0 * embedding_weights_values[0] +
                   2.0 * embedding_weights_values[1] +
                   1.0 * embedding_weights_values[4]) / 4.0,
                  embedding_weights_values[4],
                  embedding_weights_values[3],
              ],
              [
                  embedding_weights_values[2],
                  embedding_weights_values[3],
                  embedding_weights_values[3],
              ],
          ],
      )

  @test_util.run_all_in_graph_and_eager_modes
  def test_safe_embedding_lookup_sparse_3d_no_weights(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      valid_ids = np.array([0, 1, 2, -1])
      embedding_weights, embedding_weights_values, sparse_ids, _ = self._get_ids_and_weights_3d(
          valid_ids)
      embedding_lookup_result = de.safe_embedding_lookup_sparse(
          embedding_weights, sparse_ids, None)
      embedding_lookup_result = embedding_lookup_result.numpy(
      ) if context.executing_eagerly() else embedding_lookup_result.eval()
      self.assertAllClose(
          embedding_lookup_result,
          [
              [
                  (embedding_weights_values[0] + embedding_weights_values[1] +
                   embedding_weights_values[3]) / 3.0,
                  embedding_weights_values[3],
                  [0] * 4,
              ],
              [
                  embedding_weights_values[2],
                  (embedding_weights_values[0] + embedding_weights_values[1]) /
                  2.0,
                  [0] * 4,
              ],
          ],
      )

  @test_util.run_all_in_graph_and_eager_modes
  def test_safe_embedding_lookup_sparse_3d_partitioned(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      embedding_weights = _random_weights(num_shards=3)
      sparse_ids, _ = _ids_and_weights_3d()
      valid_ids = np.array([0, 1, 2, -1])

      # init
      embedding_weights_values = embedding_weights.lookup(valid_ids)
      embedding_weights_values = embedding_weights_values.numpy(
      ) if context.executing_eagerly() else embedding_weights_values.eval()
      self.evaluate(
          embedding_weights.upsert(valid_ids, embedding_weights_values))
      embedding_lookup_result = de.safe_embedding_lookup_sparse(
          embedding_weights, sparse_ids, None)
      embedding_lookup_result = embedding_lookup_result.numpy(
      ) if context.executing_eagerly() else embedding_lookup_result.eval()

      self.assertAllClose(
          embedding_lookup_result,
          [
              [
                  (embedding_weights_values[0] + embedding_weights_values[1] +
                   embedding_weights_values[3]) / 3.0,
                  embedding_weights_values[3],
                  [0] * 4,
              ],
              [
                  embedding_weights_values[2],
                  (embedding_weights_values[0] + embedding_weights_values[1]) /
                  2.0,
                  [0] * 4,
              ],
          ],
      )

  @test_util.run_all_in_graph_and_eager_modes
  def test_safe_embedding_lookup_sparse_with_initializer(self):
    id = 0
    embed_dim = 8
    dense_shape = np.array([64, 128, 32])
    total_space = 64 * 128 * 32
    elements_num = int(total_space * 0.50)
    for initializer, target_mean, target_stddev in [
        (init_ops.random_normal_initializer(0.0, 0.001), 0.0, 0.00029),
        (init_ops.truncated_normal_initializer(0.0, 0.001), 0.0, 0.00029),
        (keras_init_ops.RandomNormalV2(mean=0.0, stddev=0.001), 0.0, 0.00029),
    ]:
      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()):
        id += 1
        embedding_weights = de.get_variable(
            "safe-init-bugfix-" + str(id),
            key_dtype=dtypes.int64,
            value_dtype=dtypes.float32,
            devices=_get_devices() * 3,
            initializer=initializer,
            dim=embed_dim,
        )

        indices_1d = np.random.randint(0, total_space, elements_num)
        indices_1d = np.unique(indices_1d)
        indices_1d.sort()
        indices_3d = []
        for _i in range(indices_1d.size):
          a_indice = []
          quotient = int(indices_1d[_i] / (128 * 32))
          remainder = indices_1d[_i] % (128 * 32)
          a_indice.append(quotient)
          quotient = int(remainder / 32)
          remainder = remainder % 32
          a_indice.extend([quotient, remainder])
          indices_3d.extend([a_indice])
        indices_3d = np.array(indices_3d)

        ids = np.random.randint(
            -0x7FFFFFFFFFFFFFFF,
            0x7FFFFFFFFFFFFFFF,
            indices_1d.size,
            dtype=np.int64,
        )

        sparse_ids = sparse_tensor.SparseTensor(
            constant_op.constant(indices_3d, dtypes.int64),
            constant_op.constant(ids, dtypes.int64),
            constant_op.constant(dense_shape, dtypes.int64),
        )
        vals_op = de.safe_embedding_lookup_sparse(embedding_weights,
                                                  sparse_ids,
                                                  None,
                                                  combiner="mean")
        vals_op = vals_op.numpy() if context.executing_eagerly(
        ) else vals_op.eval()

        mean = self.evaluate(math_ops.reduce_mean(vals_op))
        stddev = self.evaluate(math_ops.reduce_std(vals_op))
        rtol = 2e-4
        atol = rtol
        self.assertTrue(not (vals_op[0][0][0] == vals_op[0][0][1]))
        self.assertAllClose(target_mean, mean, rtol, atol)
        self.assertAllClose(target_stddev, stddev, rtol, atol)

  @test_util.run_all_in_graph_and_eager_modes
  def test_safe_embedding_lookup_sparse_shape_checking(self):
    if context.executing_eagerly():
      self.skipTest("Skip eager test")
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      embed_dim = 4
      embedding_weights_nn = variable_scope.get_variable("n",
                                                         shape=[100, embed_dim],
                                                         use_resource=False)
      embedding_weights_de = _random_weights(embed_dim=4)
      sparse_ids, _ = _ids_and_weights_3d(embed_dim=embed_dim)

      embedding_lookup_base = embedding_ops.safe_embedding_lookup_sparse(
          embedding_weights_nn, sparse_ids, None)
      embedding_lookup_test = de.safe_embedding_lookup_sparse(
          embedding_weights_de, sparse_ids, None)
      self.assertAllEqual(embedding_lookup_base.shape,
                          embedding_lookup_test.shape)
      self.assertAllEqual(embedding_lookup_base.get_shape(),
                          embedding_lookup_test.get_shape())

  @test_util.run_all_in_graph_and_eager_modes
  def test_dynamic_embedding_variable_clear(self):
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

      self.evaluate(table.clear())
      self.assertAllEqual(0, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 3, 4], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertAllEqual([[-1], [-1], [-1], [-1]], result)


@test_util.run_all_in_graph_and_eager_modes
class TrainableWrapperPlacementTest(test.TestCase):

  def test_colocate_to_ids(self):
    server0 = server_lib.Server.create_local_server()
    server1 = server_lib.Server.create_local_server()
    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'dist'
    job.tasks[0] = server0.target[len('grpc://'):]
    job.tasks[1] = server1.target[len('grpc://'):]

    g = ops.Graph()

    with g.as_default():
      with ops.device('/job:dist/task:0'):
        sp_var = de.get_variable('dist414',
                                 key_dtype=dtypes.int64,
                                 value_dtype=dtypes.float32,
                                 dim=2,
                                 initializer=0.1)
      with ops.device('/job:dist/task:1'):
        features = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
        emb_p, tw_p = de.embedding_lookup(sp_var,
                                          features,
                                          name='on_task_1',
                                          return_trainable=True)
      with ops.device('/job:dist/task:0'):
        emb_q, tw_q = de.embedding_lookup(sp_var,
                                          features,
                                          name='on_task_0',
                                          return_trainable=True)
    self.assertAllEqual(tw_p.device, '/job:dist/task:1')
    self.assertAllEqual(tw_q.device, '/job:dist/task:1')


@test_util.run_all_in_graph_and_eager_modes
class EmbeddingLookupEagerTest(test.TestCase):

  def _create_input_and_params(self,
                               name,
                               batch_size=4,
                               nids=64,
                               embedding_size=1):
    assert nids % batch_size == 0
    ids = math_ops.range(0, nids, dtype=dtypes.int64)
    ids = array_ops.reshape(ids, (batch_size, -1))
    labels = array_ops.zeros((batch_size,), dtype=dtypes.float32)
    devar = de.get_variable(name + '/dynamic_embedding',
                            dim=embedding_size,
                            initializer=tf.keras.initializers.Zeros())
    tfvar = tf.Variable(tf.keras.initializers.Zeros()((nids, embedding_size),
                                                      dtype=tf.float32))
    return ids, labels, devar, tfvar

  def _loss_fn(self, params, ids, labels):

    if isinstance(params, de.Variable):
      embedding = de.embedding_lookup(params, ids)
    elif isinstance(
        params, (resource_variable_ops.ResourceVariable, variables.Variable)):
      embedding = embedding_ops.embedding_lookup(params, ids)
    else:
      raise TypeError

    logits = math_ops.reduce_mean(math_ops.reduce_sum(embedding, 1), 1)
    entropy = nn_impl.sigmoid_cross_entropy_with_logits(logits=logits,
                                                        labels=labels)
    loss = math_ops.reduce_mean(entropy)
    return loss

  def test_run_training_eagerly(self):
    if not context.executing_eagerly():
      self.skipTest('Only test functional API in eager mode.')

    batch_size = 4
    ids, labels, devar, tfvar = self._create_input_and_params('vns079',
                                                              embedding_size=1)
    nsteps = 10

    loss_fn = tf.function()(self._loss_fn)

    def sorted_dynamic_embedding_value():
      embedding_var = devar
      optimizer = Adam(1E-3)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)

      def var_fn():
        return list(embedding_var.trainable_store.values())

      for _ in range(nsteps):
        optimizer.minimize(lambda: loss_fn(embedding_var, ids, labels), var_fn)

      keys, values = embedding_var.export()
      order = tf.argsort(keys)
      return array_ops.gather(values, order)

    def sorted_static_embedding_value():
      embedding_var = tfvar
      optimizer = Adam(1E-3)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)

      def var_fn():
        return [embedding_var]

      for _ in range(nsteps):
        optimizer.minimize(lambda: loss_fn(embedding_var, ids, labels), var_fn)

      return embedding_var.read_value()

    de_values = sorted_dynamic_embedding_value()
    tf_values = sorted_static_embedding_value()
    self.assertAllClose(de_values, tf_values)


if __name__ == "__main__":
  test.main()
