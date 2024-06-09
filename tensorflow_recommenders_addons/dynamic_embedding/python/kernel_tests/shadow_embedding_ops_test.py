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
"""unit tests of embedding_lookup APIs
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import os
import tempfile
import tensorflow as tf

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import server_lib

from tensorflow_recommenders_addons.dynamic_embedding.python.ops.shadow_embedding_ops import \
  embedding_lookup_unique_base, HvdVariable
from tensorflow_recommenders_addons.utils.check_platform import is_macos, is_arm64

try:
  from tensorflow.keras.legacy.optimizers import Adam
except:
  from tensorflow.keras.optimizers import Adam


def _get_sparse_variable(name,
                         key_dtype=dtypes.int64,
                         value_dtype=dtypes.float32,
                         dim=2,
                         init_size=16,
                         ps_devices=None,
                         bp_v2=False,
                         restrict_policy=None,
                         initializer=0.1,
                         distribute_strategy=None):
  devar = de.get_variable(name,
                          key_dtype=key_dtype,
                          value_dtype=value_dtype,
                          dim=dim,
                          init_size=init_size,
                          bp_v2=bp_v2,
                          devices=ps_devices,
                          restrict_policy=restrict_policy,
                          initializer=initializer)
  shadow_name = name + '-shadow'
  shadow = de.shadow_ops.ShadowVariable(devar,
                                        name=shadow_name,
                                        distribute_strategy=distribute_strategy)
  return devar, shadow


def _sort_keys_and_values(keys, values):
  seq = np.argsort(keys)
  keys = np.sort(keys)
  values = values[seq]
  return keys, values


def _create_ps_and_worker_servers(spec):
  ps_list, worker_list = [], []
  for job_name, ip_port_list in spec.as_dict().items():
    for i, v in enumerate(ip_port_list):
      node = server_lib.Server(spec,
                               job_name=job_name,
                               task_index=i,
                               config=default_cluster_config)
      if job_name == 'ps':
        ps_list.append(node)
      elif job_name == 'worker':
        worker_list.append(node)
      else:
        raise TypeError(
            'Expecting ps or worker in cluster_spec, but get {}'.format(
                job_name))
  return ps_list, worker_list


default_config = config_pb2.ConfigProto(
    allow_soft_placement=True,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))

default_cluster_config = config_pb2.ConfigProto(allow_soft_placement=False)


@test_util.run_all_in_graph_and_eager_modes
class ShadowVariableTest(test.TestCase):

  def test_create(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    key_dtypes = [dtypes.int64]
    value_dtypes = [dtypes.int32, dtypes.float32]
    dims = [1, 4]
    trainable_options = [True, False]
    devices = ['/CPU:0']
    var_list = []
    rnd = 0
    for comb in itertools.product(key_dtypes, value_dtypes, dims,
                                  trainable_options):
      devar = de.get_variable('sparse_domain-' + str(rnd),
                              key_dtype=comb[0],
                              value_dtype=comb[1],
                              dim=comb[2],
                              initializer=0.1,
                              devices=devices,
                              init_size=1)
      name = 'shadow-' + str(rnd)
      var = de.shadow_ops.ShadowVariable(devar, name=name, trainable=comb[3])
      self.assertEqual(var.dtype, devar.value_dtype)
      self.assertEqual(var.ids.dtype, devar.key_dtype)
      rnd += 1

  def test_lookup(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      var, shadow_var = _get_sparse_variable('tk049', dim=2)
      self.evaluate(variables.global_variables_initializer())
      ids = constant_op.constant([2, 5], dtype=dtypes.int64)
      values = array_ops.ones((2, 2), dtype=np.float32)
      self.evaluate(
          var.upsert(ids, ops.convert_to_tensor(values, dtype=dtypes.float32)))

      ext_ids = constant_op.constant([2, 5, 8], dtype=dtypes.int64)
      exp_values = np.array([[1, 1], [1, 1], [0.1, 0.1]], dtype=np.float32)
      emb = self.evaluate(de.shadow_ops.embedding_lookup(shadow_var, ext_ids))
      self.assertAllEqual(exp_values, emb)

  def test_safe_embedding_lookup_sparse(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      var, shadow_var = _get_sparse_variable('tk049', dim=2)
      self.evaluate(variables.global_variables_initializer())
      ids = constant_op.constant([2, 5], dtype=dtypes.int64)
      values = array_ops.ones((2, 2), dtype=np.float32)
      self.evaluate(
          var.upsert(ids, ops.convert_to_tensor(values, dtype=dtypes.float32)))

      sp_ids = constant_op.constant([[0, 2], [1, 5]], dtype=dtypes.int64)
      sp_weights = constant_op.constant([2, 5], dtype=dtypes.int64)
      dense_shape = constant_op.constant([2, 6], dtype=dtypes.int64)
      sparse_tensor = tf.sparse.SparseTensor(indices=sp_ids,
                                             values=sp_weights,
                                             dense_shape=dense_shape)

      emb = self.evaluate(
          de.safe_embedding_lookup_sparse(shadow_var, sparse_tensor))
      self.assertAllEqual(emb, values)

  def test_hvd_safe_embedding_lookup_sparse(self):
    try:
      import horovod.tensorflow as hvd
    except Exception as e:
      self.skipTest(
          f"Skip the test for horovod import error with Tensorflow-2.7.0 on MacOS-12. {str(e)}"
      )
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')
    if (is_macos() and is_arm64()):
      self.skipTest(
          "Apple silicon devices don't support synchronous training based on Horovod."
      )
    # TODO: Resolve the conflict arising from the 'save' function incompatibility with TensorFlow 2.11.
    if (tf.__version__ == "2.11.0" or tf.__version__ == "2.11.1"):
      self.skipTest(
          "The save function doesn't work with TF 2.11, skip the test.")

    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      var, shadow_var = _get_sparse_variable('tk049', dim=2)
      hvd_var = HvdVariable("hvd_var", shadow_var, 1, mpi_size=1)
      self.evaluate(variables.global_variables_initializer())
      ids = constant_op.constant([2, 5], dtype=dtypes.int64)
      values = array_ops.ones((2, 2), dtype=np.float32)
      self.evaluate(
          var.upsert(ids, ops.convert_to_tensor(values, dtype=dtypes.float32)))

      sp_ids = constant_op.constant([[0, 2], [1, 5]], dtype=dtypes.int64)
      sp_weights = constant_op.constant([2, 5], dtype=dtypes.int64)
      dense_shape = constant_op.constant([2, 6], dtype=dtypes.int64)
      sparse_tensor = tf.sparse.SparseTensor(indices=sp_ids,
                                             values=sp_weights,
                                             dense_shape=dense_shape)

      emb = self.evaluate(
          de.safe_embedding_lookup_sparse(hvd_var, sparse_tensor))
      self.assertAllEqual(emb, values)

  def test_update_with_optimizer_v1(self):
    if not context.executing_eagerly():
      self.skipTest('Only test when eagerly.')

    for bp_v2 in [False, True]:
      var, shadow_var = _get_sparse_variable('bh890-bpv2-%s' % bp_v2,
                                             dim=2,
                                             bp_v2=bp_v2)
      optimizer = adam.AdamOptimizer(1E-3)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)
      initialized = False
      with self.session(use_gpu=test_util.is_gpu_available(),
                        config=default_config):
        ids = []
        for i in range(10):
          ids.append(i)
          tf_ids = ops.convert_to_tensor(ids, dtype=dtypes.int64)

          def _loss_fn(shadow_var, ids):
            emb = de.shadow_ops.embedding_lookup(
                shadow_var, ops.convert_to_tensor(ids, dtype=dtypes.int64))
            loss = math_ops.reduce_mean(emb, axis=0)
            loss = array_ops.reshape(loss, (-1, 2))
            loss = math_ops.matmul(loss, array_ops.transpose(loss))
            return loss

          train_op = optimizer.minimize(lambda: _loss_fn(shadow_var, ids),
                                        var_list=[shadow_var])
          if not initialized:
            self.evaluate(variables.global_variables_initializer())
            initialized = True
          self.evaluate(train_op)
          keys, values = _sort_keys_and_values(*self.evaluate(var.export()))
          result_keys, result_values = _sort_keys_and_values(*self.evaluate(
              [shadow_var.ids, shadow_var.read_value(False)]))
          self.assertAllEqual(keys, result_keys)
          self.assertAllEqual(values, result_values)

  def test_update_with_optimizer_v2(self):
    if not context.executing_eagerly():
      self.skipTest('Only test when eagerly.')

    for bp_v2 in [False, True]:
      var, shadow_var = _get_sparse_variable('bh890-bpv2-%s' % bp_v2,
                                             dim=2,
                                             bp_v2=bp_v2)
      optimizer = optimizer_v2.adagrad.Adagrad(1E-3)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)
      initialized = False
      with self.session(use_gpu=test_util.is_gpu_available(),
                        config=default_config):
        ids = []
        for i in range(10):
          ids.append(i)
          tf_ids = ops.convert_to_tensor(ids, dtype=dtypes.int64)

          def _loss_fn(shadow_var, ids):
            emb = de.shadow_ops.embedding_lookup(
                shadow_var, ops.convert_to_tensor(ids, dtype=dtypes.int64))
            loss = math_ops.reduce_mean(emb, axis=0)
            loss = array_ops.reshape(loss, (-1, 2))
            loss = math_ops.matmul(loss, array_ops.transpose(loss))
            return loss

          train_op = optimizer.minimize(lambda: _loss_fn(shadow_var, ids),
                                        [shadow_var])
          if not initialized:
            self.evaluate(variables.global_variables_initializer())
            initialized = True
          self.evaluate(train_op)
          keys, values = _sort_keys_and_values(*self.evaluate(var.export()))
          result_keys, result_values = _sort_keys_and_values(*self.evaluate(
              [shadow_var.ids, shadow_var.read_value(False)]))
          self.assertAllEqual(keys, result_keys)
          self.assertAllEqual(values, result_values)

  def test_wrapper_tf_function(self):
    if not context.executing_eagerly():
      self.skipTest('Skip test tf.function in eager mode.')
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      var, shadow_var = _get_sparse_variable('pf988', dim=2)
      optimizer = optimizer_v2.adam.Adam(1E-4)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)

      @def_function.function
      def compute_fn(var, ids):
        emb = de.shadow_ops.embedding_lookup(var, ids)
        return math_ops.reduce_mean(emb)

      start = 0
      size = 0
      for i in range(10):
        ids = math_ops.range(start, i + 1, dtype=dtypes.int64)
        start = math_ops.reduce_max(ids) + 1
        size += array_ops.size(ids)
        optimizer.minimize(lambda: compute_fn(shadow_var, ids), [shadow_var])
        self.assertAllEqual(var.size(), size)

  def test_training_with_restrict_policy(self):
    if not context.executing_eagerly():
      self.skipTest('Skip test tf.function in eager mode.')

    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      var, shadow_var = _get_sparse_variable(
          'pf988', dim=2, restrict_policy=de.TimestampRestrictPolicy)
      optimizer = optimizer_v2.adam.Adam(1E-4)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)

      @def_function.function
      def compute_fn(var, ids):
        emb = de.shadow_ops.embedding_lookup(var, ids)
        return math_ops.reduce_mean(emb)

      start = 0
      size = 0
      for i in range(10):
        ids = math_ops.range(start, i + 1, dtype=dtypes.int64)
        start = math_ops.reduce_max(ids) + 1
        size += array_ops.size(ids)
        optimizer.minimize(lambda: compute_fn(shadow_var, ids), [shadow_var])
        self.assertAllEqual(var.size(), size)
        self.assertAllEqual(var.restrict_policy.status.size(), size)

  def test_training_with_distributed_strategy(self):
    # TODO(Lifann) Servers will be alive and thus make other test cases
    # across the cases failed. So this case is kept only for demonstration.
    self.skipTest('Only for demonstration.')

    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    cluster_spec = tf.train.ClusterSpec({
        'ps': ['localhost:2220', 'localhost:2221'],
        'worker': ['localhost:2222', 'localhost:2223']
    })
    ps_list, worker_list = _create_ps_and_worker_servers(cluster_spec)

    resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec)
    strategy = tf.distribute.experimental.ParameterServerStrategy(resolver)
    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
        strategy)
    with strategy.scope() as scope:
      var = de.get_variable('pf988',
                            dim=2,
                            initializer=0.1,
                            devices=['/job:ps/task:0', '/job:ps/task:1'])
      shadow_var = de.shadow_ops.ShadowVariable(var,
                                                name='pf988-shadow',
                                                distribute_strategy=strategy)
      optimizer = optimizer_v2.adam.Adam(1E-4)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)

    def dist_dataset_fn():
      dataset_values = np.arange(0, 10, dtype=np.int64)
      fn = lambda x: tf.data.Dataset.from_tensor_slices(dataset_values).batch(
          4).repeat(None)
      return strategy.distribute_datasets_from_function(fn)

    dataset = coordinator.create_per_worker_dataset(dist_dataset_fn)

    @tf.function
    def step_fn(iterator):

      def replica_fn(ids):

        def loss_fn(ids):
          batch_size = tf.shape(ids)[0]
          emb = de.shadow_ops.embedding_lookup(shadow_var, ids)
          loss = tf.reduce_mean(emb)
          return loss

        optimizer.minimize(lambda: loss_fn(ids), [shadow_var])

      return strategy.run(replica_fn, args=(next(iterator),))

    iterator = iter(dataset)
    for i in range(5):
      coordinator.schedule(step_fn, args=(iterator,))
    coordinator.join()
    self.assertAllEqual(var.size(), 10)


@test_util.run_all_in_graph_and_eager_modes
class ShadowVariableBasicBehaviorTest(test.TestCase):

  def test_read_value(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    params = de.get_variable('pn012', dim=2, initializer=0.1)
    params.upsert(
        constant_op.constant([1, 2, 3], dtype=dtypes.int64),
        constant_op.constant([[1., 1.], [2., 2.], [3., 3.]],
                             dtype=dtypes.float32))
    shadow = de.shadow_ops.ShadowVariable(params)

    val = shadow.read_value()
    self.assertAllEqual(val.numpy().tolist(), [])

    ids = constant_op.constant([2, 3, 4], dtype=dtypes.int64)
    shadow._reset_ids(ids)
    val = gen_resource_variable_ops.read_variable_op(shadow._handle,
                                                     dtypes.float32)
    self.assertAllEqual(val.numpy().tolist(), [])
    val = shadow.read_value(do_prefetch=False)
    self.assertAllEqual(val.numpy().tolist(), [])
    val = shadow.read_value(do_prefetch=True)
    self.assertAllEqual(
        val,
        constant_op.constant([[2., 2.], [3., 3.], [0.1, 0.1]],
                             dtype=dtypes.float32))

    ids = constant_op.constant([3, 4, 5], dtype=dtypes.int64)
    shadow._reset_ids(ids)
    val = shadow.read_value(do_prefetch=False)
    self.assertAllEqual(
        val,
        constant_op.constant([[2., 2.], [3., 3.], [0.1, 0.1]],
                             dtype=dtypes.float32))
    val = shadow.read_value(do_prefetch=True)
    self.assertAllEqual(
        val,
        constant_op.constant([[3., 3.], [0.1, 0.1], [0.1, 0.1]],
                             dtype=dtypes.float32))

  def test_embedding_lookup(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    params = de.get_variable('pn012', dim=2, initializer=0.1)
    params.upsert(
        constant_op.constant([1, 2, 3], dtype=dtypes.int64),
        constant_op.constant([[1., 1.], [2., 2.], [3., 3.]],
                             dtype=dtypes.float32))
    shadow = de.shadow_ops.ShadowVariable(params)

    ids = constant_op.constant([2, 3, 4], dtype=dtypes.int64)
    val = de.shadow_ops.embedding_lookup(shadow, ids)
    self.assertAllEqual(
        val,
        constant_op.constant([[2., 2.], [3., 3.], [0.1, 0.1]],
                             dtype=dtypes.float32))

    params.upsert(
        constant_op.constant([1, 2, 3], dtype=dtypes.int64),
        constant_op.constant([[1.1, 1.1], [2.2, 2.2], [3.3, 3.3]],
                             dtype=dtypes.float32))
    val = de.shadow_ops.embedding_lookup(shadow, ids)
    self.assertAllEqual(
        val,
        constant_op.constant([[2.2, 2.2], [3.3, 3.3], [0.1, 0.1]],
                             dtype=dtypes.float32))

  def test_embedding_lookup_unique(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    params = de.get_variable('pn012', dim=2, initializer=0.1)
    params.upsert(
        constant_op.constant([1, 2, 3], dtype=dtypes.int64),
        constant_op.constant([[1., 1.], [2., 2.], [3., 3.]],
                             dtype=dtypes.float32))
    shadow = de.shadow_ops.ShadowVariable(params)

    ids_tensor = tf.constant([[2, 3], [4, 5], [1, 0]], dtype=tf.int64)

    val_tensor = de.shadow_ops.embedding_lookup_unique(shadow, ids_tensor, 2,
                                                       True)

    expected_output = tf.constant([
        [[2., 2.], [3., 3.]],  # embeddings for ids 2 and 3
        [[0.1, 0.1], [0.1,
                      0.1]],  # embeddings for ids 4 and 5 (default initialized)
        [[1., 1.], [0.1,
                    0.1]]  # embeddings for id 1 and 0 (default initialized)
    ])
    self.assertAllEqual(val_tensor, expected_output)

  def test_embedding_lookup_unique_hvd(self):
    if not tf.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    params = tf.Variable(
        [[0.1, 0.1], [1., 1.], [2., 2.], [3., 3.], [0.1, 0.1], [0.1, 0.1]],
        dtype=tf.float32)

    ids_tensor = tf.constant([[2, 3], [4, 5], [1, 0]], dtype=tf.int64)

    val_tensor = embedding_lookup_unique_base(
        ids_tensor, 2, lambda ids: tf.gather(params, ids), True,
        "mock_embedding")

    expected_output = tf.constant([
        [[2., 2.], [3., 3.]],  # embeddings for ids 2 and 3
        [[0.1, 0.1], [0.1,
                      0.1]],  # embeddings for ids 4 and 5 (default initialized)
        [[1., 1.], [0.1,
                    0.1]]  # embeddings for id 1 and 0 (default initialized)
    ])

    self.assertAllEqual(val_tensor, expected_output)

  def test_ragged_embedding_lookup_unique_hvd(self):
    if not tf.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    params = tf.Variable(
        [[0.1, 0.1], [1., 1.], [2., 2.], [3., 3.], [0.1, 0.1], [0.1, 0.1]],
        dtype=tf.float32)

    ragged_ids = tf.RaggedTensor.from_row_splits(values=tf.constant(
        [2, 3, 4, 5, 1], dtype=tf.int64),
                                                 row_splits=[0, 2, 5])
    val_ragged_tensor = embedding_lookup_unique_base(
        ragged_ids, 2, lambda ids: tf.gather(params, ids), True,
        "mock_embedding")

    expected_output = tf.RaggedTensor.from_row_splits(values=tf.constant(
        [[2., 2.], [3., 3.], [0.1, 0.1], [0.1, 0.1], [1., 1.]],
        dtype=tf.float32),
                                                      row_splits=[0, 2, 5])

    self.assertAllEqual(val_ragged_tensor.to_tensor(),
                        expected_output.to_tensor())

  def test_ragged_embedding_lookup_unique(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    params = de.get_variable('pn012', dim=2, initializer=0.1)
    params.upsert(
        constant_op.constant([1, 2, 3], dtype=dtypes.int64),
        constant_op.constant([[1., 1.], [2., 2.], [3., 3.]],
                             dtype=dtypes.float32))
    shadow = de.shadow_ops.ShadowVariable(params)
    # [[2, 3], [4, 5, 1]]
    ragged_ids = tf.RaggedTensor.from_row_splits(values=tf.constant(
        [2, 3, 4, 5, 1], dtype=tf.int64),
                                                 row_splits=[0, 2, 5])
    val_ragged = de.shadow_ops.embedding_lookup_unique(shadow, ragged_ids, 2,
                                                       True)
    expected_output = tf.RaggedTensor.from_row_splits(values=[[2., 2.],
                                                              [3., 3.],
                                                              [0.1, 0.1],
                                                              [0.1, 0.1],
                                                              [1., 1.]],
                                                      row_splits=[0, 2, 5])
    self.assertAllEqual(val_ragged, expected_output)

  def test_get_size(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    params = de.get_variable('pn012', dim=2, initializer=0.1)
    params.upsert(
        constant_op.constant([1, 2, 3], dtype=dtypes.int64),
        constant_op.constant([[1., 1.], [2., 2.], [3., 3.]],
                             dtype=dtypes.float32))
    shadow = de.shadow_ops.ShadowVariable(params)
    self.assertEqual(shadow.size(), 3)

  def test_update_by_optimizer(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    sparse_params = de.get_variable('pn012', dim=2, initializer=0.0)
    sparse_params.upsert(
        constant_op.constant([1, 2, 3], dtype=dtypes.int64),
        constant_op.constant([[1., 1.], [2., 2.], [3., 3.]],
                             dtype=dtypes.float32))
    shadow = de.shadow_ops.ShadowVariable(sparse_params)
    dense_params = variables.Variable([[1., 1.], [2., 2.], [3., 3.]],
                                      dtype=dtypes.float32)

    sparse_optimizer = Adam(1E-3)
    sparse_optimizer = de.DynamicEmbeddingOptimizer(sparse_optimizer)
    dense_optimizer = Adam(1E-3)
    dense_optimizer = de.DynamicEmbeddingOptimizer(dense_optimizer)

    def sparse_loss():
      ids = constant_op.constant([1, 3], dtype=dtypes.int64)
      emb = de.shadow_ops.embedding_lookup(shadow, ids)
      return math_ops.reduce_mean(emb)

    def dense_loss():
      ids = constant_op.constant([0, 2], dtype=dtypes.int64)
      emb = array_ops.gather(dense_params, ids)
      return math_ops.reduce_mean(emb)

    for i in range(10):
      sparse_optimizer.minimize(sparse_loss, var_list=[shadow])
      dense_optimizer.minimize(dense_loss, var_list=[dense_params])

    ids = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
    values = sparse_params.lookup(ids)
    self.assertAllClose(values, dense_params)

    sparse_slot_params = sparse_params.get_slot_variables(sparse_optimizer)
    if hasattr(dense_optimizer, 'get_slot_names'):
      dense_slot_params = [
          dense_optimizer.get_slot(dense_params, name)
          for name in dense_optimizer.get_slot_names()
      ]
    else:
      dense_slot_params = [
          s for s in dense_optimizer._variables if 'iteration' not in s.name
      ]

    for i in range(len(sparse_slot_params)):
      sparse_values = sparse_slot_params[i].lookup(ids)
      dense_values = dense_slot_params[i]
      self.assertAllClose(sparse_values, dense_values)

  def test_update_by_optimizer_bpv2(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    sparse_params = de.get_variable('kr193', dim=2, initializer=0.0, bp_v2=True)
    sparse_params.upsert(
        constant_op.constant([1, 2, 3], dtype=dtypes.int64),
        constant_op.constant([[2.4, 3.1], [5.1, -0.7], [-15.2, 3.9]],
                             dtype=dtypes.float32))
    shadow = de.shadow_ops.ShadowVariable(sparse_params)
    dense_params = variables.Variable([[2.4, 3.1], [5.1, -0.7], [-15.2, 3.9]],
                                      dtype=dtypes.float32)

    sparse_optimizer = Adam(1E-4)
    sparse_optimizer = de.DynamicEmbeddingOptimizer(sparse_optimizer)
    dense_optimizer = Adam(1E-4)
    dense_optimizer = de.DynamicEmbeddingOptimizer(dense_optimizer)

    rtol = 2e-4
    atol = 2e-6

    def sparse_loss():
      ids = constant_op.constant([1, 3], dtype=dtypes.int64)
      emb = de.shadow_ops.embedding_lookup(shadow, ids)
      return math_ops.reduce_mean(emb)

    def dense_loss():
      ids = constant_op.constant([0, 2], dtype=dtypes.int64)
      emb = array_ops.gather(dense_params, ids)
      return math_ops.reduce_mean(emb)

    for i in range(10):
      sparse_optimizer.minimize(sparse_loss, var_list=[shadow])
      dense_optimizer.minimize(dense_loss, var_list=[dense_params])

    ids = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
    values = sparse_params.lookup(ids)
    self.assertAllClose(values, dense_params, rtol, atol)

    sparse_slot_params = sparse_params.get_slot_variables(sparse_optimizer)
    if hasattr(dense_optimizer, 'get_slot_names'):
      dense_slot_params = [
          dense_optimizer.get_slot(dense_params, name)
          for name in dense_optimizer.get_slot_names()
      ]
    else:
      dense_slot_params = [
          s for s in dense_optimizer._variables if 'iteration' not in s.name
      ]

    for i in range(len(sparse_slot_params)):
      sparse_values = sparse_slot_params[i].lookup(ids)
      dense_values = dense_slot_params[i]
      self.assertAllClose(sparse_values, dense_values)

  def test_save_and_restore_with_trace(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    class TestModule(tf.Module):

      def __init__(self):
        self.params = de.get_variable('nb910',
                                      dim=2,
                                      initializer=0.0,
                                      bp_v2=True)
        self.shadow = de.shadow_ops.ShadowVariable(self.params)

      def __call__(self, x):
        embed = de.shadow_ops.embedding_lookup(self.shadow, x)
        #return math_ops.reduce_mean(embed)
        return embed

      def size(self):
        return self.params.size()

    module = TestModule()
    keys = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
    values = constant_op.constant([[2.4, 3.1], [5.1, -0.7], [-15.2, 3.9]],
                                  dtype=dtypes.float32)
    module.params.upsert(keys, values)
    module(keys)
    self.assertAllEqual(module.shadow.read_value(False), values)

    model_dir = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save_ckpt_dir = os.path.join(model_dir, 'model')

    options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
    ckpt = de.train.DECheckpoint(module)
    ckpt.save(save_ckpt_dir)
    shadow_value = module.shadow.read_value(False)
    self.assertAllEqual(shadow_value.shape, (0, 2))  # clear when saving

    new_module = TestModule()
    new_ckpt = de.train.DECheckpoint(new_module)
    restore_ckpt_path = tf.train.latest_checkpoint(model_dir)
    new_ckpt.read(restore_ckpt_path)
    self.assertEqual(new_module.size(), 3)
    expected_values = module(keys)
    self.assertAllEqual(expected_values, values)

    shadow_value = new_module.shadow.read_value(False)
    self.assertAllEqual(shadow_value.shape, (0, 2))

  def test_save_and_restore_with_trace_file_system(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    class TestModule(tf.Module):

      def __init__(self, test_devices):
        self.params = de.get_variable(
            'nb910fs',
            devices=test_devices,
            dim=1,
            initializer=0.0,
            bp_v2=True,
            kv_creator=de.CuckooHashTableCreator(saver=de.FileSystemSaver()))
        self.shadow = de.shadow_ops.ShadowVariable(self.params)

      def __call__(self, x):
        embed = de.shadow_ops.embedding_lookup(self.shadow, x)
        return embed

      def size(self):
        return self.params.size()

    class TestNewModule(tf.Module):

      def __init__(self, test_devices):
        self.params = de.get_variable(
            'nb910fs',
            devices=test_devices,
            dim=1,
            initializer=0.0,
            bp_v2=True,
            kv_creator=de.CuckooHashTableCreator(saver=de.FileSystemSaver()))
        self.shadow = de.shadow_ops.ShadowVariable(self.params)
        self.dense = variables.Variable([[2.4, 3.1], [5.1, -0.7], [-15.2, 3.9]],
                                        dtype=dtypes.float32,
                                        name='test_var')

      def __call__(self, x):
        embed = de.shadow_ops.embedding_lookup(self.shadow, x)
        return embed

      def size(self):
        return self.params.size()

    test_size = 10
    test_keys = [i for i in range(0, test_size)]
    test_values = [[i * 1.0] for i in range(0, test_size)]
    table_device = ['/device:CPU:0']
    if test_util.is_gpu_available():
      table_device = ['/device:GPU:0']
    shard_num = 3
    table_devices_ = table_device * shard_num
    module = TestModule(table_devices_)
    keys = constant_op.constant(test_keys, dtype=dtypes.int64)
    values = constant_op.constant(test_values, dtype=dtypes.float32)
    module.params.upsert(keys, values)
    module(keys)
    self.assertAllEqual(module.shadow.read_value(False), values)

    model_dir = tempfile.mkdtemp(prefix=self.get_temp_dir())
    save_ckpt_dir = os.path.join(model_dir, 'model')

    options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
    ckpt = de.train.DECheckpoint(module)
    ckpt.save(save_ckpt_dir)
    shadow_value = module.shadow.read_value(False)
    self.assertAllEqual(shadow_value.shape, (0, 1))  # clear when saving

    tf.keras.backend.clear_session()
    del module, ckpt
    new_module = TestNewModule(table_devices_)
    new_ckpt = de.train.DECheckpoint(new_module)
    restore_ckpt_path = tf.train.latest_checkpoint(model_dir)
    new_ckpt.read(restore_ckpt_path)
    self.assertEqual(new_module.size(), test_size)
    expected_values = new_module(keys)
    self.assertAllEqual(np.sort(expected_values, axis=0), values)

    # test expand shards number
    tf.keras.backend.clear_session()
    del new_module, new_ckpt
    shard_num = 5
    table_devices_ = table_device * shard_num
    new_module = TestNewModule(table_devices_)
    new_ckpt = de.train.DECheckpoint(new_module)
    new_ckpt.read(restore_ckpt_path)
    self.assertEqual(new_module.size(), test_size)
    expected_values = new_module(keys)
    self.assertAllEqual(np.sort(expected_values, axis=0), values)

    # test contracte shards number
    tf.keras.backend.clear_session()
    del new_module, new_ckpt
    shard_num = 2
    table_devices_ = table_device * shard_num
    new_module = TestNewModule(table_devices_)
    new_ckpt = de.train.DECheckpoint(new_module)
    new_ckpt.read(restore_ckpt_path)
    self.assertEqual(new_module.size(), test_size)
    expected_values = new_module(keys)
    self.assertAllEqual(np.sort(expected_values, axis=0), values)

    # test load all into one shard
    tf.keras.backend.clear_session()
    del new_module, new_ckpt
    shard_num = 1
    table_devices_ = table_device * shard_num
    new_module = TestNewModule(table_devices_)
    new_ckpt = de.train.DECheckpoint(new_module)
    new_ckpt.read(restore_ckpt_path)
    self.assertEqual(new_module.size(), test_size)
    expected_values = new_module(keys)
    self.assertAllEqual(np.sort(expected_values, axis=0), values)


if __name__ == '__main__':
  test.main()
