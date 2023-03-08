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
"""unit tests of dynamic embedding ops
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import itertools
import numpy as np
import os
import tensorflow as tf

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.keras import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adadelta
from tensorflow.python.training import adagrad
from tensorflow.python.training import adagrad_da
from tensorflow.python.training import adam
from tensorflow.python.training import ftrl
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import momentum
from tensorflow.python.training import monitored_session
from tensorflow.python.training import proximal_adagrad
from tensorflow.python.training import proximal_gradient_descent as pgd
from tensorflow.python.training import rmsprop
from tensorflow.python.training import training_util


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


default_config = config_pb2.ConfigProto(
    allow_soft_placement=True,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))


class CommonTrainableTestV1Base(object):

  def common_minimize_trainable(self, base_opt, test_opt, name, bp_v2):
    raise NotImplementedError

  def device_check(self, de):
    if test_util.is_gpu_available():
      self.assertTrue("GPU" in de.tables[0].resource_handle.device.upper())

  @test_util.deprecated_graph_mode_only
  def test_adadelta_minimize_trainable(self):
    base_opt = adadelta.AdadeltaOptimizer(1.0)
    test_opt = adadelta.AdadeltaOptimizer(1.0)
    self.common_minimize_trainable(base_opt,
                                   test_opt,
                                   name="adadelta",
                                   bp_v2=False)

  @test_util.deprecated_graph_mode_only
  def test_adagrad_minimize_trainable(self):
    base_opt = adagrad.AdagradOptimizer(1.0)
    test_opt = adagrad.AdagradOptimizer(1.0)
    self.common_minimize_trainable(base_opt,
                                   test_opt,
                                   name="adagrad",
                                   bp_v2=False)

  @test_util.deprecated_graph_mode_only
  def test_adagradda_minimize_trainable(self):
    base_gs = training_util.create_global_step()

    base_opt = adagrad_da.AdagradDAOptimizer(1.0, base_gs)
    test_opt = adagrad_da.AdagradDAOptimizer(1.0, base_gs)
    self.common_minimize_trainable(base_opt,
                                   test_opt,
                                   name="adagrad_da",
                                   bp_v2=False)

  @test_util.deprecated_graph_mode_only
  def test_ftrl_minimize_trainable(self):
    base_opt = ftrl.FtrlOptimizer(1.0)
    test_opt = ftrl.FtrlOptimizer(1.0)
    self.common_minimize_trainable(base_opt, test_opt, name="ftrl", bp_v2=False)

  @test_util.deprecated_graph_mode_only
  def test_proximal_adagrad_minimize_trainable(self):
    base_opt = proximal_adagrad.ProximalAdagradOptimizer(1.0)
    test_opt = proximal_adagrad.ProximalAdagradOptimizer(1.0)
    self.common_minimize_trainable(base_opt,
                                   test_opt,
                                   name="proximal_adagrad",
                                   bp_v2=False)

  @test_util.deprecated_graph_mode_only
  def test_proximalsgd_minimize_trainable(self):
    base_opt = pgd.ProximalGradientDescentOptimizer(1.0)
    test_opt = pgd.ProximalGradientDescentOptimizer(1.0)
    self.common_minimize_trainable(base_opt,
                                   test_opt,
                                   name="proximal_sgd",
                                   bp_v2=False)

  @test_util.deprecated_graph_mode_only
  def test_momentum_minimize_trainable(self):
    base_opt = momentum.MomentumOptimizer(1.0, momentum=0.9)
    test_opt = momentum.MomentumOptimizer(1.0, momentum=0.9)
    self.common_minimize_trainable(base_opt,
                                   test_opt,
                                   name="momentum",
                                   bp_v2=False)

  @test_util.deprecated_graph_mode_only
  def test_sgd_minimize_trainable(self):
    base_opt = gradient_descent.GradientDescentOptimizer(1.0)
    test_opt = gradient_descent.GradientDescentOptimizer(1.0)
    self.common_minimize_trainable(base_opt, test_opt, name="sgd", bp_v2=False)

  @test_util.deprecated_graph_mode_only
  def test_adam_minimize_trainable(self):
    base_opt = adam.AdamOptimizer(1.0)
    test_opt = adam.AdamOptimizer(1.0)
    self.common_minimize_trainable(base_opt, test_opt, name="adam", bp_v2=False)

  @test_util.deprecated_graph_mode_only
  def test_rmsprop_minimize_trainable(self):
    for centered_ in [False, True]:
      base_opt = rmsprop.RMSPropOptimizer(1.0, centered=centered_)
      test_opt = rmsprop.RMSPropOptimizer(1.0, centered=centered_)
      self.common_minimize_trainable(base_opt,
                                     test_opt,
                                     name="rmsprop" + str(centered_),
                                     bp_v2=False)

  @test_util.deprecated_graph_mode_only
  def test_adadelta_minimize_trainable_bpv2(self):
    base_opt = adadelta.AdadeltaOptimizer(1.0)
    test_opt = adadelta.AdadeltaOptimizer(1.0)
    self.common_minimize_trainable(base_opt,
                                   test_opt,
                                   name="adadelta",
                                   bp_v2=True)

  @test_util.deprecated_graph_mode_only
  def test_adagrad_minimize_trainable_bpv2(self):
    base_opt = adagrad.AdagradOptimizer(1.0)
    test_opt = adagrad.AdagradOptimizer(1.0)
    self.common_minimize_trainable(base_opt,
                                   test_opt,
                                   name="adagrad",
                                   bp_v2=True)

  @test_util.deprecated_graph_mode_only
  def test_adagradda_minimize_trainable_bpv2(self):
    base_gs = training_util.create_global_step()

    base_opt = adagrad_da.AdagradDAOptimizer(1.0, base_gs)
    test_opt = adagrad_da.AdagradDAOptimizer(1.0, base_gs)
    self.common_minimize_trainable(base_opt,
                                   test_opt,
                                   name="adagrad_da",
                                   bp_v2=True)

  @test_util.deprecated_graph_mode_only
  def test_ftrl_minimize_trainable_bpv2(self):
    base_opt = ftrl.FtrlOptimizer(1.0)
    test_opt = ftrl.FtrlOptimizer(1.0)
    self.common_minimize_trainable(base_opt, test_opt, name="ftrl", bp_v2=True)

  @test_util.deprecated_graph_mode_only
  def test_proximal_adagrad_minimize_trainable_bpv2(self):
    base_opt = proximal_adagrad.ProximalAdagradOptimizer(1.0)
    test_opt = proximal_adagrad.ProximalAdagradOptimizer(1.0)
    self.common_minimize_trainable(base_opt,
                                   test_opt,
                                   name="proximal_adagrad",
                                   bp_v2=True)

  @test_util.deprecated_graph_mode_only
  def test_proximalsgd_minimize_trainable_bpv2(self):
    base_opt = pgd.ProximalGradientDescentOptimizer(1.0)
    test_opt = pgd.ProximalGradientDescentOptimizer(1.0)
    self.common_minimize_trainable(base_opt,
                                   test_opt,
                                   name="proximal_sgd",
                                   bp_v2=True)

  @test_util.deprecated_graph_mode_only
  def test_momentum_minimize_trainable_bpv2(self):
    base_opt = momentum.MomentumOptimizer(1.0, momentum=0.9)
    test_opt = momentum.MomentumOptimizer(1.0, momentum=0.9)
    self.common_minimize_trainable(base_opt,
                                   test_opt,
                                   name="momentum",
                                   bp_v2=True)

  @test_util.deprecated_graph_mode_only
  def test_sgd_minimize_trainable_bpv2(self):
    base_opt = gradient_descent.GradientDescentOptimizer(1.0)
    test_opt = gradient_descent.GradientDescentOptimizer(1.0)
    self.common_minimize_trainable(base_opt, test_opt, name="sgd", bp_v2=True)

  @test_util.deprecated_graph_mode_only
  def test_adam_minimize_trainable_bpv2(self):
    base_opt = adam.AdamOptimizer(1.0)
    test_opt = adam.AdamOptimizer(1.0)
    self.common_minimize_trainable(base_opt, test_opt, name="adam", bp_v2=True)

  @test_util.deprecated_graph_mode_only
  def test_rmsprop_minimize_trainable_bpv2(self):
    for centered_ in [False, True]:
      base_opt = rmsprop.RMSPropOptimizer(1.0, centered=centered_)
      test_opt = rmsprop.RMSPropOptimizer(1.0, centered=centered_)
      self.common_minimize_trainable(base_opt,
                                     test_opt,
                                     name="rmsprop" + str(centered_),
                                     bp_v2=True)


class CommonTrainableTestV2Base(object):

  def common_minimize_trainable_v2(self, base_opt, test_opt, name):
    raise NotImplementedError

  def device_check(self, de):
    if test_util.is_gpu_available():
      self.assertTrue("GPU" in de.tables[0].resource_handle.device.upper())

  @test_util.run_in_graph_and_eager_modes
  def test_adadelta_v2_minimize_trainable(self):
    if test_util.is_gpu_available():
      self.skipTest("Skip GPU Test for no GPU kernel.")
    base_opt = optimizer_v2.adadelta.Adadelta(1.0)
    test_opt = optimizer_v2.adadelta.Adadelta(1.0)
    self.common_minimize_trainable_v2(base_opt, test_opt, name="adadelta")

  @test_util.run_in_graph_and_eager_modes
  def test_adagrad_v2_minimize_trainable(self):
    if test_util.is_gpu_available():
      self.skipTest("Skip GPU Test for no GPU kernel.")
    base_opt = optimizer_v2.adagrad.Adagrad(1.0)
    test_opt = optimizer_v2.adagrad.Adagrad(1.0)
    self.common_minimize_trainable_v2(base_opt, test_opt, name="adagrad")

  @test_util.run_in_graph_and_eager_modes
  def test_adam_v2_minimize_trainable(self):
    base_opt = optimizer_v2.adam.Adam(1.0)
    test_opt = optimizer_v2.adam.Adam(1.0)
    self.common_minimize_trainable_v2(base_opt, test_opt, name="adam")

  @test_util.run_in_graph_and_eager_modes
  def test_adamax_v2_minimize_trainable(self):
    if test_util.is_gpu_available():
      self.skipTest("Skip GPU Test for GPU kernel has bug.")
    base_opt = optimizer_v2.adamax.Adamax(1.0)
    test_opt = optimizer_v2.adamax.Adamax(1.0)
    self.common_minimize_trainable_v2(base_opt, test_opt, name="adamax")

  @test_util.run_in_graph_and_eager_modes
  def test_ftrl_v2_minimize_trainable(self):
    if test_util.is_gpu_available():
      self.skipTest("Skip GPU Test for no GPU kernel.")
    base_opt = optimizer_v2.ftrl.Ftrl(1.0)
    test_opt = optimizer_v2.ftrl.Ftrl(1.0)
    self.common_minimize_trainable_v2(base_opt, test_opt, name="ftrl")

  @test_util.run_in_graph_and_eager_modes
  def test_sgd_v2_minimize_trainable(self):
    base_opt = optimizer_v2.gradient_descent.SGD(1.0)
    test_opt = optimizer_v2.gradient_descent.SGD(1.0)
    self.common_minimize_trainable_v2(base_opt, test_opt, name="sgd")

  @test_util.run_in_graph_and_eager_modes
  def test_nadam_v2_minimize_trainable(self):
    base_opt = optimizer_v2.nadam.Nadam(1.0)
    test_opt = optimizer_v2.nadam.Nadam(1.0)
    self.common_minimize_trainable_v2(base_opt, test_opt, name="Nadam")

  @test_util.run_in_graph_and_eager_modes
  def test_rmsprop_v2_minimize_trainable(self):
    base_opt = optimizer_v2.rmsprop.RMSprop(1.0)
    test_opt = optimizer_v2.rmsprop.RMSprop(1.0)
    self.common_minimize_trainable_v2(base_opt, test_opt, name="rmsprop")


class EmbeddingLookupTrainableV1Test(test.TestCase, CommonTrainableTestV1Base):

  def common_minimize_trainable(self, base_opt, test_opt, name, bp_v2):
    de.enable_train_mode()
    base_opt = de.DynamicEmbeddingOptimizer(base_opt)
    test_opt = de.DynamicEmbeddingOptimizer(test_opt)
    id = 0
    for (
        num_shards,
        k_dtype,
        d_dtype,
        initial_mode,
        dim,
        run_step,
    ) in itertools.product(
        [1, 2],
        [
            dtypes.int64,
        ],
        [
            dtypes.float32,
        ],
        [
            "constant",
        ],
        [1, 10],
        [10],
    ):
      id += 1
      with self.session(use_gpu=test_util.is_gpu_available(),
                        config=default_config) as sess:
        # common define
        raw_init_ids = [0, 1]
        raw_init_vals = np.random.rand(2, dim)
        raw_ids = [
            0,
        ]
        x = constant_op.constant(np.random.rand(dim, len(raw_ids)),
                                 dtype=d_dtype)

        # base graph
        base_var = resource_variable_ops.ResourceVariable(raw_init_vals,
                                                          dtype=d_dtype)
        ids = constant_op.constant(raw_ids, dtype=k_dtype)
        pred0 = math_ops.matmul(
            embedding_ops.embedding_lookup([base_var], ids, name='by3050'), x)
        loss0 = pred0 * pred0
        base_opt_op = base_opt.minimize(loss0)

        # test graph
        embeddings = de.get_variable(
            "t2020-" + name + str(id),
            key_dtype=k_dtype,
            value_dtype=d_dtype,
            devices=_get_devices() * num_shards,
            initializer=1.0,
            dim=dim,
            bp_v2=bp_v2,
        )
        self.device_check(embeddings)
        init_ids = constant_op.constant(raw_init_ids, dtype=k_dtype)
        init_vals = constant_op.constant(raw_init_vals, dtype=d_dtype)
        init_op = embeddings.upsert(init_ids, init_vals)
        self.evaluate(init_op)

        test_var, trainable = de.embedding_lookup([embeddings],
                                                  ids,
                                                  return_trainable=True,
                                                  name='bp1256')
        pred1 = math_ops.matmul(test_var, x)
        loss1 = pred1 * pred1

        test_opt_op = test_opt.minimize(loss1, var_list=[trainable])

        self.evaluate(variables.global_variables_initializer())

        for _ in range(run_step):
          sess.run(base_opt_op)

        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType(raw_init_vals[raw_ids],
                                           self.evaluate(test_var))
        # Run `run_step` step of sgd
        for _ in range(run_step):
          sess.run(test_opt_op)

        table_var = embeddings.lookup(ids)
        # Validate updated params
        self.assertAllCloseAccordingToType(
            self.evaluate(base_var)[raw_ids],
            self.evaluate(table_var),
            msg="Cond:{},{},{},{},{},{}".format(num_shards, k_dtype, d_dtype,
                                                initial_mode, dim, run_step),
        )


class EmbeddingLookupTrainableV2Test(test.TestCase, CommonTrainableTestV2Base):

  def common_minimize_trainable_v2(self, base_opt, test_opt, name):
    de.enable_train_mode()
    tf.config.set_soft_device_placement(True)
    base_opt = de.DynamicEmbeddingOptimizer(base_opt)
    test_opt = de.DynamicEmbeddingOptimizer(test_opt)
    id = 0
    for (
        num_shards,
        k_dtype,
        d_dtype,
        initial_mode,
        dim,
        run_step,
    ) in itertools.product(
        [1, 2],
        [
            dtypes.int64,
        ],
        [
            dtypes.float32,
        ],
        [
            "constant",
        ],
        [1, 10],
        [10],
    ):
      id += 1
      # common define
      raw_init_ids = [0, 1]
      raw_init_vals = np.random.rand(2, dim)
      raw_ids = [
          0,
      ]

      # base graph
      def base_fn():
        with ops.Graph().as_default(), self.cached_session():
          embeddings = resource_variable_ops.ResourceVariable(raw_init_vals,
                                                              dtype=d_dtype)

          def loss_fn(emb):
            ids = constant_op.constant(raw_ids, dtype=k_dtype)
            pred = embedding_ops.embedding_lookup([emb], ids, name='ct9143')
            return pred * pred

          base_opt_op = base_opt.minimize(lambda: loss_fn(embeddings),
                                          [embeddings])
          self.evaluate(variables.global_variables_initializer())
          for _ in range(run_step):
            self.evaluate(base_opt_op)
          return self.evaluate(embeddings)

      base_opt_val = base_fn()

      def test_fn():
        with ops.Graph().as_default(), self.cached_session():
          embeddings = de.get_variable(
              "t2020-v2-" + name + str(id),
              key_dtype=k_dtype,
              value_dtype=d_dtype,
              devices=_get_devices() * num_shards,
              initializer=1.0,
              dim=dim,
          )
          self.device_check(embeddings)
          trainables = []
          init_ids = constant_op.constant(raw_init_ids, dtype=k_dtype)
          init_vals = constant_op.constant(raw_init_vals, dtype=d_dtype)
          self.evaluate(embeddings.upsert(init_ids, init_vals))

          def var_fn():
            return trainables

          def loss_fn(x, trainables):
            ids = constant_op.constant(raw_ids, dtype=k_dtype)
            pred, trainable = de.embedding_lookup([x],
                                                  ids,
                                                  return_trainable=True,
                                                  name='xg5785')
            trainables.clear()
            trainables.append(trainable)
            return pred * pred

          test_opt_op = test_opt.minimize(
              lambda: loss_fn(embeddings, trainables), var_fn)
          self.evaluate(variables.global_variables_initializer())
          for _ in range(run_step):
            self.evaluate(test_opt_op)
          return self.evaluate(embeddings.lookup(init_ids))

      with ops.device(_get_devices()[0]):
        test_opt_val = test_fn()
      self.assertAllCloseAccordingToType(
          base_opt_val,
          test_opt_val,
          msg="Cond:{},{},{},{},{},{}".format(num_shards, k_dtype, d_dtype,
                                              initial_mode, dim, run_step),
      )


class EmbeddingLookupUniqueTrainableV1Test(test.TestCase,
                                           CommonTrainableTestV1Base):

  def common_minimize_trainable(self, base_opt, test_opt, name, bp_v2):
    de.enable_train_mode()
    base_opt = de.DynamicEmbeddingOptimizer(base_opt)
    test_opt = de.DynamicEmbeddingOptimizer(test_opt)
    id = 0
    for (
        num_shards,
        k_dtype,
        d_dtype,
        initial_mode,
        dim,
        run_step,
    ) in itertools.product(
        [1, 2],
        [
            dtypes.int64,
        ],
        [
            dtypes.float32,
        ],
        [
            "constant",
        ],
        [1, 10],
        [10],
    ):
      id += 1
      with self.session(use_gpu=test_util.is_gpu_available(),
                        config=default_config) as sess:
        # common define
        raw_init_ids = [0, 1, 2, 3, 4]
        raw_init_vals = np.random.rand(5, dim)
        raw_ids = [0, 1, 1, 2, 3, 4, 4]
        x = constant_op.constant(np.random.rand(dim, len(raw_ids)),
                                 dtype=d_dtype)

        # base graph
        ids = constant_op.constant(raw_ids, dtype=k_dtype)
        base_var = resource_variable_ops.ResourceVariable(raw_init_vals,
                                                          dtype=d_dtype)
        unique_ids, idx = array_ops.unique(ids)
        unique_embeddings = embedding_ops.embedding_lookup([base_var],
                                                           unique_ids,
                                                           name='rt4647')
        embeddings = array_ops.gather(unique_embeddings, idx)
        pred0 = math_ops.matmul(embeddings, x)
        loss0 = pred0 * pred0
        base_opt_op = base_opt.minimize(loss0)

        # test graph
        embeddings = de.get_variable(
            "t-embedding_lookup_unique-v1-" + name + str(id),
            key_dtype=k_dtype,
            value_dtype=d_dtype,
            devices=_get_devices() * num_shards,
            initializer=1.0,
            dim=dim,
        )
        self.device_check(embeddings)
        init_ids = constant_op.constant(raw_init_ids, dtype=k_dtype)
        init_vals = constant_op.constant(raw_init_vals, dtype=d_dtype)
        init_op = embeddings.upsert(init_ids, init_vals)
        self.evaluate(init_op)

        test_var, trainable = de.embedding_lookup_unique([embeddings],
                                                         ids,
                                                         return_trainable=True,
                                                         name='bf1117')
        pred1 = math_ops.matmul(test_var, x)
        loss1 = pred1 * pred1

        test_opt_op = test_opt.minimize(loss1, var_list=[trainable])

        self.evaluate(variables.global_variables_initializer())

        for _ in range(run_step):
          sess.run(base_opt_op)

        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType(raw_init_vals[raw_ids],
                                           self.evaluate(test_var))
        # Run `run_step` step of sgd
        for _ in range(run_step):
          sess.run(test_opt_op)

        table_var = embeddings.lookup(ids)
        # Validate updated params
        self.assertAllCloseAccordingToType(
            self.evaluate(base_var)[raw_ids],
            self.evaluate(table_var),
            msg="Cond:{},{},{},{},{},{}".format(num_shards, k_dtype, d_dtype,
                                                initial_mode, dim, run_step),
        )


class EmbeddingLookupUniqueTrainableV2Test(test.TestCase,
                                           CommonTrainableTestV2Base):

  def common_minimize_trainable_v2(self, base_opt, test_opt, name):
    de.enable_train_mode()
    base_opt = de.DynamicEmbeddingOptimizer(base_opt)
    test_opt = de.DynamicEmbeddingOptimizer(test_opt)
    id = 0
    for (
        num_shards,
        k_dtype,
        d_dtype,
        initial_mode,
        dim,
        run_step,
    ) in itertools.product(
        [1, 2],
        [
            dtypes.int64,
        ],
        [
            dtypes.float32,
        ],
        [
            "constant",
        ],
        [1, 10],
        [10],
    ):
      id += 1
      # common define
      raw_init_ids = [0, 1, 2, 3, 4]
      raw_init_vals = np.random.rand(5, dim)
      raw_ids = [0, 1, 1, 2, 3, 4, 4]

      # base graph
      def base_fn():
        embeddings = resource_variable_ops.ResourceVariable(raw_init_vals,
                                                            dtype=d_dtype)

        def loss_fn(emb):
          ids = constant_op.constant(raw_ids, dtype=k_dtype)
          unique_ids, idx = array_ops.unique(ids)
          unique_embeddings = embedding_ops.embedding_lookup([emb],
                                                             unique_ids,
                                                             name='sa6943')
          pred = array_ops.gather(unique_embeddings, idx)
          return pred * pred

        base_opt_op = base_opt.minimize(lambda: loss_fn(embeddings),
                                        [embeddings])
        self.evaluate(variables.global_variables_initializer())
        for _ in range(run_step):
          self.evaluate(base_opt_op)
        return embeddings

      base_opt_val = self.evaluate(base_fn())

      def test_fn():
        embeddings = de.get_variable(
            "t2020-v2-" + name + str(id),
            key_dtype=k_dtype,
            value_dtype=d_dtype,
            devices=_get_devices() * num_shards,
            initializer=1.0,
            dim=dim,
        )
        self.device_check(embeddings)
        trainables = []
        init_ids = constant_op.constant(raw_init_ids, dtype=k_dtype)
        init_vals = constant_op.constant(raw_init_vals, dtype=d_dtype)
        self.evaluate(embeddings.upsert(init_ids, init_vals))

        def var_fn():
          return trainables

        def loss_fn(x, trainables):
          ids = constant_op.constant(raw_ids, dtype=k_dtype)
          pred, trainable = de.embedding_lookup_unique([x],
                                                       ids,
                                                       return_trainable=True,
                                                       name='ff8889')
          trainables.clear()
          trainables.append(trainable)
          return pred * pred

        test_opt_op = test_opt.minimize(lambda: loss_fn(embeddings, trainables),
                                        var_fn)
        self.evaluate(variables.global_variables_initializer())
        for _ in range(run_step):
          self.evaluate(test_opt_op)
        return embeddings.lookup(init_ids)

      with ops.device(_get_devices()[0]):
        test_opt_val = self.evaluate(test_fn())
      self.assertAllCloseAccordingToType(
          base_opt_val,
          test_opt_val,
          msg="Cond:{},{},{},{},{},{}".format(num_shards, k_dtype, d_dtype,
                                              initial_mode, dim, run_step),
      )


class EmbeddingLookupSparseTrainableV1Test(test.TestCase,
                                           CommonTrainableTestV1Base):

  def common_minimize_trainable(self, base_opt, test_opt, name, bp_v2):
    de.enable_train_mode()
    base_opt = de.DynamicEmbeddingOptimizer(base_opt)
    test_opt = de.DynamicEmbeddingOptimizer(test_opt)
    id = 0
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = False
    for (
        num_shards,
        k_dtype,
        d_dtype,
        initial_mode,
        dim,
        run_step,
    ) in itertools.product(
        [1, 2],
        [dtypes.int64],
        [
            dtypes.float32,
        ],
        [
            "constant",
        ],
        [1, 10],
        [10],
    ):
      id += 1
      raw_init_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      raw_init_vals = [[
          x,
      ] * dim for x in [0.0, 0.1, 0.3, 0.8, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81]]

      raw_ids = constant_op.constant([1, 3, 3, 9], dtype=k_dtype)
      sp_ids = sparse_tensor.SparseTensor(
          indices=[
              [0, 0],
              [0, 1],
              [1, 0],
              [2, 1],
          ],
          values=raw_ids,
          dense_shape=[3, 2],
      )
      x = constant_op.constant([[_x * dim] for _x in [[0.4], [0.5], [0.6]]],
                               dtype=d_dtype)

      x = array_ops.reshape(x, shape=(3 * dim, 1))
      # base branch
      with self.session(use_gpu=test_util.is_gpu_available(),
                        config=default_config) as sess:
        base_var = variables.Variable(
            np.array(raw_init_vals).reshape([len(raw_init_ids), dim]),
            dtype=d_dtype,
            shape=[len(raw_init_ids), dim],
        )
        base_embedding = embedding_ops.embedding_lookup_sparse(base_var,
                                                               sp_ids,
                                                               None,
                                                               combiner="sum",
                                                               name='wg8689')
        base_embedding = array_ops.reshape(base_embedding, shape=[1, 3 * dim])
        pred0 = math_ops.matmul(base_embedding, x)
        loss0 = pred0 * pred0

        base_opt_op = base_opt.minimize(loss0, var_list=[base_var])
        # run base
        self.evaluate(variables.global_variables_initializer())
        for _ in range(run_step):
          sess.run(base_opt_op)

        base_var_val = self.evaluate(base_var)

      # test branch
      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        # test var prepare
        embeddings = de.get_variable(
            "t1030-" + name + str(id),
            key_dtype=k_dtype,
            value_dtype=d_dtype,
            devices=_get_devices() * num_shards,
            initializer=1.0,
            dim=dim,
        )
        self.device_check(embeddings)

        init_ids = constant_op.constant(raw_init_ids, dtype=k_dtype)
        init_vals = constant_op.constant(raw_init_vals, dtype=d_dtype)
        init_op = embeddings.upsert(init_ids, init_vals)
        self.evaluate(init_op)

        test_var, trainable = de.embedding_lookup_sparse(
            embeddings,
            sp_ids,
            sp_weights=None,
            combiner="sum",
            return_trainable=True,
            name='io3441',
        )

        pred1 = math_ops.matmul(array_ops.reshape(test_var, shape=[1, 3 * dim]),
                                x)
        loss1 = pred1 * pred1
        test_opt_op = test_opt.minimize(loss1, var_list=[trainable])

        self.evaluate(variables.global_variables_initializer())

        self.assertAllCloseAccordingToType(
            np.array(raw_init_vals).reshape([len(raw_init_ids), dim]),
            self.evaluate(base_var),
        )

        # Run `run_step` step of sgd
        for _ in range(run_step):
          sess.run(test_opt_op)
        if test_util.is_gpu_available():
          self.assertTrue(
              _check_device(embeddings.tables[0].resource_handle, "GPU"))

        table_var_val = self.evaluate(
            array_ops.reshape(embeddings.lookup(init_ids), shape=[10, dim]))
      # Validate updated params
      self.assertAllCloseAccordingToType(
          base_var_val,
          table_var_val,
          msg="Cond:{},{},{},{},{}".format(num_shards, k_dtype, d_dtype, dim,
                                           run_step),
      )


class EmbeddingLookupSparseTrainableV2Test(test.TestCase,
                                           CommonTrainableTestV2Base):

  def common_minimize_trainable_v2(self, base_opt, test_opt, name):
    de.enable_train_mode()
    tf.config.set_soft_device_placement(True)
    base_opt = de.DynamicEmbeddingOptimizer(base_opt)
    test_opt = de.DynamicEmbeddingOptimizer(test_opt)
    id = 0
    for (
        num_shards,
        k_dtype,
        d_dtype,
        initial_mode,
        dim,
        run_step,
    ) in itertools.product(
        [1, 2],
        [
            dtypes.int64,
        ],
        [
            dtypes.float32,
        ],
        [
            "constant",
        ],
        [1, 10],
        [10],
    ):
      id += 1
      raw_init_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      raw_init_vals = [[
          x,
      ] * dim for x in [0.0, 0.1, 0.3, 0.8, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81]]
      with ops.device(_get_devices()[0]):
        raw_ids = constant_op.constant([1, 3, 3, 9], dtype=k_dtype)
        sp_ids = sparse_tensor.SparseTensor(
            indices=[
                [0, 0],
                [0, 1],
                [1, 0],
                [2, 1],
            ],
            values=raw_ids,
            dense_shape=[3, 2],
        )
        x = constant_op.constant([[_x * dim] for _x in [[0.4], [0.5], [0.6]]],
                                 dtype=d_dtype)
        x = array_ops.reshape(x, shape=(dim, -1))

      # # base graph
      def base_fn():
        embeddings = variables.Variable(
            np.array(raw_init_vals).reshape([len(raw_init_ids), dim]),
            dtype=d_dtype,
            shape=[len(raw_init_ids), dim],
        )

        def loss_fn(emb):
          embedding = embedding_ops.embedding_lookup_sparse(emb,
                                                            sp_ids,
                                                            None,
                                                            combiner="sum",
                                                            name='gc4675')
          pred = math_ops.matmul(embedding, x)
          return pred * pred

        base_opt_op = base_opt.minimize(lambda: loss_fn(embeddings),
                                        [embeddings])
        self.evaluate(variables.global_variables_initializer())
        for _ in range(run_step):
          self.evaluate(base_opt_op)
        return embeddings

      base_opt_val = self.evaluate(base_fn())

      def test_fn():
        embeddings = de.get_variable(
            "t1030-v2-" + name + str(id),
            key_dtype=k_dtype,
            value_dtype=d_dtype,
            devices=_get_devices() * num_shards,
            initializer=1.0,
            dim=dim,
        )
        self.device_check(embeddings)

        init_ids = constant_op.constant(raw_init_ids, dtype=k_dtype)
        init_vals = constant_op.constant(raw_init_vals, dtype=d_dtype)
        self.evaluate(embeddings.upsert(init_ids, init_vals))
        trainables = []

        def var_fn():
          return trainables

        def loss_fn(emb, trainables):
          test_var, trainable = de.embedding_lookup_sparse(
              emb,
              sp_ids,
              sp_weights=None,
              combiner="sum",
              return_trainable=True,
              name='bk3382',
          )

          pred = math_ops.matmul(test_var, x)
          trainables.clear()
          trainables.append(trainable)
          return pred * pred

        test_opt_op = test_opt.minimize(lambda: loss_fn(embeddings, trainables),
                                        var_fn)
        self.evaluate(variables.global_variables_initializer())
        for _ in range(run_step):
          self.evaluate(test_opt_op)
        return embeddings.lookup(init_ids)

      test_opt_val = self.evaluate(test_fn())
      self.assertAllCloseAccordingToType(
          base_opt_val,
          test_opt_val,
          msg="Cond:{},{},{},{},{},{}".format(num_shards, k_dtype, d_dtype,
                                              initial_mode, dim, run_step),
      )


class SafeEmbeddingLookupSparseTrainableV1Test(test.TestCase,
                                               CommonTrainableTestV1Base):

  @test_util.deprecated_graph_mode_only
  def common_minimize_trainable(self, base_opt, test_opt, name, bp_v2):
    de.enable_train_mode()
    base_opt = de.DynamicEmbeddingOptimizer(base_opt)
    test_opt = de.DynamicEmbeddingOptimizer(test_opt)
    id = 0
    config = config_pb2.ConfigProto(
        allow_soft_placement=True,
        gpu_options=config_pb2.GPUOptions(allow_growth=True),
    )
    for (
        num_shards,
        k_dtype,
        d_dtype,
        initial_mode,
        dim,
        run_step,
    ) in itertools.product(
        [1, 2],
        [dtypes.int64],
        [
            dtypes.float32,
        ],
        [
            "constant",
        ],
        [1, 10],
        [10],
    ):
      with self.session(config=config, use_gpu=test_util.is_gpu_available()):
        id += 1
        raw_init_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        raw_init_vals = [
            [
                x,
            ] * dim
            for x in [0.0, 0.1, 0.3, 0.8, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81]
        ]
        raw_ids = constant_op.constant([1, 3, 3, 9], dtype=k_dtype)
        sp_ids = sparse_tensor.SparseTensor(
            indices=[
                [0, 0],
                [0, 1],
                [1, 0],
                [2, 1],
            ],
            values=raw_ids,
            dense_shape=[3, 2],
        )
        x = constant_op.constant([[_x * dim] for _x in [[0.4], [0.5], [0.6]]],
                                 dtype=d_dtype)
        x = array_ops.reshape(x, shape=(3 * dim, 1))
        # base var prepare
        base_var = variables.Variable(
            np.array(raw_init_vals).reshape([len(raw_init_ids), dim]),
            dtype=d_dtype,
            shape=[len(raw_init_ids), dim],
        )
        base_embedding = embedding_ops.safe_embedding_lookup_sparse(
            base_var, sp_ids, None, combiner="sum", name='zp0198')
        base_embedding = array_ops.reshape(base_embedding, shape=[1, 3 * dim])
        pred0 = math_ops.matmul(base_embedding, x)
        loss0 = pred0 * pred0

        base_opt_op = base_opt.minimize(loss0, var_list=[base_var])

        # test var prepare
        embeddings = de.get_variable(
            "s6030-" + name + str(id),
            key_dtype=k_dtype,
            value_dtype=d_dtype,
            devices=_get_devices() * num_shards,
            initializer=1.0,
            dim=dim,
        )
        self.device_check(embeddings)

        init_ids = constant_op.constant(raw_init_ids, dtype=k_dtype)
        init_vals = constant_op.constant(raw_init_vals, dtype=d_dtype)
        init_op = embeddings.upsert(init_ids, init_vals)
        self.evaluate(init_op)

        # test branch
        test_var, trainable = de.safe_embedding_lookup_sparse(
            embeddings,
            sp_ids,
            sparse_weights=None,
            combiner="sum",
            return_trainable=True,
            name='js2442',
        )

        pred1 = math_ops.matmul(array_ops.reshape(test_var, shape=[1, 3 * dim]),
                                x)
        loss1 = pred1 * pred1
        test_opt_op = test_opt.minimize(loss1, var_list=[trainable])

        self.evaluate(variables.global_variables_initializer())

        self.assertAllCloseAccordingToType(
            np.array(raw_init_vals).reshape([len(raw_init_ids), dim]),
            self.evaluate(base_var),
        )

        # run base
        for _ in range(run_step):
          self.evaluate(base_opt_op)

        # Run `run_step` step of sgd
        for _ in range(run_step):
          self.evaluate(test_opt_op)

        table_var = array_ops.reshape(embeddings.lookup(init_ids),
                                      shape=[10, dim])
        # Validate updated params
        self.assertAllCloseAccordingToType(
            self.evaluate(base_var),
            self.evaluate(table_var),
            msg="Cond:{},{},{},{},{}".format(num_shards, k_dtype, d_dtype, dim,
                                             run_step),
        )


class SafeEmbeddingLookupSparseTrainableV2Test(test.TestCase,
                                               CommonTrainableTestV2Base):

  def common_minimize_trainable_v2(self, base_opt, test_opt, name):
    de.enable_train_mode()
    tf.config.set_soft_device_placement(True)
    base_opt = de.DynamicEmbeddingOptimizer(base_opt)
    test_opt = de.DynamicEmbeddingOptimizer(test_opt)
    id = 0
    for (
        num_shards,
        k_dtype,
        d_dtype,
        initial_mode,
        dim,
        run_step,
    ) in itertools.product(
        [1, 2],
        [
            dtypes.int64,
        ],
        [
            dtypes.float32,
        ],
        [
            "constant",
        ],
        [1, 10],
        [10],
    ):
      id += 1
      raw_init_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      raw_init_vals = [[
          x,
      ] * dim for x in [0.0, 0.1, 0.3, 0.8, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81]]
      raw_ids = constant_op.constant([1, 3, 3, 9], dtype=k_dtype)
      sp_ids = sparse_tensor.SparseTensor(
          indices=[
              [0, 0],
              [0, 1],
              [1, 0],
              [2, 1],
          ],
          values=raw_ids,
          dense_shape=[3, 2],
      )
      x = constant_op.constant([[_x * dim] for _x in [[0.4], [0.5], [0.6]]],
                               dtype=d_dtype)
      x = array_ops.reshape(x, shape=(dim, -1))

      # # base graph
      def base_fn():
        embeddings = variables.Variable(
            np.array(raw_init_vals).reshape([len(raw_init_ids), dim]),
            dtype=d_dtype,
            shape=[len(raw_init_ids), dim],
        )

        def loss_fn(emb):
          embedding = embedding_ops.safe_embedding_lookup_sparse(
              emb,
              sp_ids,
              None,
              combiner="sum",
              name='uw5859',
          )
          pred0 = math_ops.matmul(embedding, x)
          return pred0 * pred0

        base_opt_op = base_opt.minimize(lambda: loss_fn(embeddings),
                                        [embeddings])
        self.evaluate(variables.global_variables_initializer())
        for _ in range(run_step):
          self.evaluate(base_opt_op)
        return embeddings

      base_opt_val = self.evaluate(base_fn())

      def test_fn():
        embeddings = de.get_variable(
            "s6030-v2-" + name + str(id),
            key_dtype=k_dtype,
            value_dtype=d_dtype,
            devices=_get_devices() * num_shards,
            initializer=1.0,
            dim=dim,
        )
        self.device_check(embeddings)

        init_ids = constant_op.constant(raw_init_ids, dtype=k_dtype)
        init_vals = constant_op.constant(raw_init_vals, dtype=d_dtype)
        self.evaluate(embeddings.upsert(init_ids, init_vals))
        trainables = []

        def var_fn():
          return trainables

        def loss_fn(emb, trainables):
          test_var, trainable = de.safe_embedding_lookup_sparse(
              emb,
              sp_ids,
              sparse_weights=None,
              combiner="sum",
              return_trainable=True,
              name='mm7462',
          )

          pred = math_ops.matmul(test_var, x)
          trainables.clear()
          trainables.append(trainable)
          return pred * pred

        test_opt_op = test_opt.minimize(lambda: loss_fn(embeddings, trainables),
                                        var_fn)
        self.evaluate(variables.global_variables_initializer())
        for _ in range(run_step):
          self.evaluate(test_opt_op)
        return embeddings.lookup(init_ids)

      test_opt_val = test_fn()
      self.assertAllCloseAccordingToType(
          base_opt_val,
          test_opt_val,
          msg="Cond:{},{},{},{},{},{}".format(num_shards, k_dtype, d_dtype,
                                              initial_mode, dim, run_step),
      )


@test_util.deprecated_graph_mode_only
class TrainDynamicEmbeddingInMonitoredTrainingSessionTest(test.TestCase):
  """Tests Training in MonitoredTrainingSession."""

  def device_check(self, de):
    if test_util.is_gpu_available():
      self.assertTrue("GPU" in de.tables[0].resource_handle.device.upper())

  def test_saving_restoring_checkpoint(self):

    logdir = _test_dir(self.get_temp_dir(), "test_saving_restoring_checkpoint")
    with ops.Graph().as_default():
      gstep = training_util.create_global_step()
      do_step = state_ops.assign_add(gstep, 1)

      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      target_values = [[0.0], [1.0], [2.0]]
      keys = array_ops.placeholder(dtypes.int64)
      values = constant_op.constant(target_values, dtypes.float32)

      table = de.Variable(
          key_dtype=dtypes.int64,
          value_dtype=dtypes.float32,
          initializer=-1.0,
          name="m100",
          dim=1,
      )
      upsert_op = table.upsert(keys, values)
      lookup_op = table.lookup(keys)
      size_op = table.size()
      with monitored_session.MonitoredTrainingSession(
          config=default_config, is_chief=True, checkpoint_dir=logdir) as sess:
        self.assertEqual(0, sess.run(gstep))
        self.assertEqual(1, sess.run(do_step))
        self.assertEqual(2, sess.run(do_step))

        # Check that the parameter nodes have been initialized.
        self.assertEqual(10.0, sess.run(v0))
        self.assertEqual(20.0, sess.run(v1))
        self.assertAllEqual(0, sess.run(size_op))
        sess.run(upsert_op, feed_dict={keys: [0, 1, 2]})
        self.assertAllEqual(3, sess.run(size_op))
        self.device_check(table)

      # A restart will find the checkpoint and recover automatically.
      with monitored_session.MonitoredTrainingSession(
          config=default_config, is_chief=True, checkpoint_dir=logdir) as sess:
        self.assertEqual(2, sess.run(gstep))
        self.assertAllEqual(3, sess.run(table.size()))
        self.assertAllEqual(target_values,
                            sess.run(lookup_op, feed_dict={keys: [0, 1, 2]}))

        self.device_check(table)

  def common_minimize_trainable(self, base_opt, test_opt, name, bp_v2):
    de.enable_train_mode()
    base_opt = de.DynamicEmbeddingOptimizer(base_opt)
    test_opt = de.DynamicEmbeddingOptimizer(test_opt)
    id = 0
    for (
        num_shards,
        k_dtype,
        d_dtype,
        initial_mode,
        dim,
        run_step,
    ) in itertools.product(
        [3],
        [dtypes.int64],
        [
            dtypes.float32,
        ],
        [
            "constant",
        ],
        [1, 10],
        [10],
    ):
      with ops.Graph().as_default():
        id += 1
        raw_init_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        raw_init_vals = [
            [
                x,
            ] * dim
            for x in [0.0, 0.1, 0.3, 0.8, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81]
        ]
        raw_ids = constant_op.constant([1, 3, 3, 9], dtype=k_dtype)
        sp_ids = sparse_tensor.SparseTensor(
            indices=[
                [0, 0],
                [0, 1],
                [1, 0],
                [2, 1],
            ],
            values=raw_ids,
            dense_shape=[3, 2],
        )
        x = constant_op.constant([[_x * dim] for _x in [[0.4], [0.5], [0.6]]],
                                 dtype=d_dtype)
        x = array_ops.reshape(x, shape=(3 * dim, 1))
        # base var prepare
        base_var = variables.Variable(
            np.array(raw_init_vals).reshape([len(raw_init_ids), dim]),
            dtype=d_dtype,
            shape=[len(raw_init_ids), dim],
        )

        # test var prepare
        embeddings = de.get_variable(
            "t1030-" + name + str(id),
            key_dtype=k_dtype,
            value_dtype=d_dtype,
            devices=_get_devices() * num_shards,
            initializer=1.0,
            dim=dim,
        )

        init_ids = constant_op.constant(raw_init_ids, dtype=k_dtype)
        init_vals = constant_op.constant(raw_init_vals, dtype=d_dtype)
        init_op = embeddings.upsert(init_ids, init_vals)

        # base branch
        base_embedding = embedding_ops.embedding_lookup_sparse(base_var,
                                                               sp_ids,
                                                               None,
                                                               combiner="sum",
                                                               name='tg2722')
        base_embedding = array_ops.reshape(base_embedding, shape=[1, 3 * dim])
        pred0 = math_ops.matmul(base_embedding, x)
        loss0 = pred0 * pred0

        base_opt_op = base_opt.minimize(loss0, var_list=[base_var])

        # test branch
        test_var, trainable = de.embedding_lookup_sparse(
            embeddings,
            sp_ids,
            sp_weights=None,
            combiner="sum",
            return_trainable=True,
            name='rq3232',
        )

        pred1 = math_ops.matmul(array_ops.reshape(test_var, shape=[1, 3 * dim]),
                                x)
        loss1 = pred1 * pred1

        gstep = training_util.create_global_step()
        test_opt_op = test_opt.minimize(loss1,
                                        var_list=[trainable],
                                        global_step=gstep)

        table_var = array_ops.reshape(embeddings.lookup(init_ids),
                                      shape=[10, dim])

        with monitored_session.MonitoredTrainingSession(
            is_chief=True, config=default_config) as sess:
          sess.run(init_op)
          self.assertAllCloseAccordingToType(
              np.array(raw_init_vals).reshape([len(raw_init_ids), dim]),
              sess.run(base_var),
          )

          # run base
          for _ in range(run_step):
            sess.run(base_opt_op)
            sess.run(test_opt_op)

          # Validate global_step
          self.assertEqual(run_step, sess.run(gstep))

          # Validate updated params
          self.assertAllCloseAccordingToType(
              sess.run(base_var),
              sess.run(table_var),
              msg="Cond:{},{},{},{},{}".format(num_shards, k_dtype, d_dtype,
                                               dim, run_step),
          )
          self.device_check(embeddings)

  def test_adam_minimize_trainable(self):
    base_opt = adam.AdamOptimizer(0.1)
    test_opt = adam.AdamOptimizer(0.1)
    self.common_minimize_trainable(base_opt, test_opt, name="adam", bp_v2=False)

  def test_adagrad_minimize_trainable(self):
    base_opt = adagrad.AdagradOptimizer(0.1)
    test_opt = adagrad.AdagradOptimizer(0.1)
    self.common_minimize_trainable(base_opt,
                                   test_opt,
                                   name="adagrad",
                                   bp_v2=False)

  def test_adam_minimize_trainable_bp_v2(self):
    base_opt = adam.AdamOptimizer(0.1)
    test_opt = adam.AdamOptimizer(0.1)
    self.common_minimize_trainable(base_opt, test_opt, name="adam", bp_v2=True)

  def test_adagrad_minimize_trainable_bp_v2(self):
    base_opt = adagrad.AdagradOptimizer(0.1)
    test_opt = adagrad.AdagradOptimizer(0.1)
    self.common_minimize_trainable(base_opt,
                                   test_opt,
                                   name="adagrad",
                                   bp_v2=True)


@test_util.deprecated_graph_mode_only
class ModelModeTest(test.TestCase):
  """Tests ModelMode."""

  def test_check_ops_number(self):
    self.assertTrue(de.get_model_mode() == "train")
    de.enable_inference_mode()
    self.assertTrue(de.get_model_mode() == "inference")
    de.enable_train_mode()
    self.assertTrue(de.get_model_mode() == "train")
    for fn, assign_num, read_num in [(de.enable_train_mode, 1, 2),
                                     (de.enable_inference_mode, 0, 1)]:
      fn()
      embeddings = de.get_variable('ModeModeTest' + str(assign_num),
                                   key_dtype=dtypes.int64,
                                   value_dtype=dtypes.float32,
                                   devices=_get_devices(),
                                   initializer=1.,
                                   dim=8)
      ids = constant_op.constant([0, 1, 2, 3, 4], dtype=dtypes.int64)
      test_var, trainable = de.embedding_lookup([embeddings],
                                                ids,
                                                return_trainable=True,
                                                name='mc4009')
      _ = math_ops.add(test_var, 1)
      op_list = ops.get_default_graph().get_operations()
      op_list_assign = [
          op.name for op in op_list if "AssignBeforeReadVariable" in op.name
      ]
      op_list_read = [op.name for op in op_list if "ReadVariableOp" in op.name]
      self.assertTrue(len(op_list_assign) == assign_num)
      self.assertTrue(len(op_list_read) == read_num)
      de.enable_train_mode()
      ops.reset_default_graph()

  def test_inference_numberic_correctness(self):
    train_pred = None
    infer_pred = None
    dim = 8
    initializer = init_ops.random_normal_initializer(0.0, 0.001)
    raw_init_vals = np.random.rand(100, dim)

    for fn in [de.enable_train_mode, de.enable_inference_mode]:
      with ops.Graph().as_default():
        fn()

        init_ids = constant_op.constant(list(range(100)), dtype=dtypes.int64)
        init_vals = constant_op.constant(raw_init_vals, dtype=dtypes.float32)
        with variable_scope.variable_scope("modelmode",
                                           reuse=variable_scope.AUTO_REUSE):
          embeddings = de.get_variable('ModelModeTest-numberic',
                                       key_dtype=dtypes.int64,
                                       value_dtype=dtypes.float32,
                                       devices=_get_devices() * 2,
                                       initializer=initializer,
                                       dim=dim)

          w = variables.Variable(1.0, name="w")
          _ = training_util.create_global_step()
        init_op = embeddings.upsert(init_ids, init_vals)

        ids = constant_op.constant([0, 1, 2, 3, 4], dtype=dtypes.int64)
        test_var, trainable = de.embedding_lookup([embeddings],
                                                  ids,
                                                  return_trainable=True,
                                                  name='xt9595')
        pred = math_ops.add(test_var, 1) * w
        loss = pred * pred
        opt = de.DynamicEmbeddingOptimizer(adagrad.AdagradOptimizer(0.1))
        opt.minimize(loss)

        with monitored_session.MonitoredTrainingSession(
            is_chief=True, config=default_config) as sess:
          if de.get_model_mode() == de.ModelMode.TRAIN:
            sess.run(init_op)
            train_pred = sess.run(pred)
          elif de.get_model_mode() == de.ModelMode.INFERENCE:
            sess.run(init_op)
            infer_pred = sess.run(pred)
      de.enable_train_mode()
      ops.reset_default_graph()
    self.assertAllEqual(train_pred, infer_pred)


if __name__ == "__main__":
  test.main()
