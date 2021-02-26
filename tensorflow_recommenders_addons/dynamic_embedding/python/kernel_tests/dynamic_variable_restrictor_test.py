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
"""Unit tests of dynamic variable restrictor"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tempfile
import time

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import layers
from tensorflow.python.keras import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adadelta
from tensorflow.python.training import adagrad
from tensorflow.python.training import adagrad_da
from tensorflow.python.training import adam
from tensorflow.python.training import ftrl
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import momentum
from tensorflow.python.training import proximal_adagrad
from tensorflow.python.training import proximal_gradient_descent as pgd
from tensorflow.python.training import rmsprop
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import server_lib
from tensorflow.python.training import training_util


def select_slot_vars(trainable_wrappers, optmz):
  slot_names = optmz.get_slot_names()
  slot_vars = []
  for name in slot_names:
    for trainable in trainable_wrappers:
      slot = optmz.get_slot(trainable, name)
      if slot is not None:
        slot_vars.append(slot.params)
  return slot_vars


def simple_embedding(var, ids):
  x = constant_op.constant(np.random.rand(2, 3), dtype=dtypes.float32)
  embed_w, trainable = de.embedding_lookup(var, ids, return_trainable=True)
  return embed_w, trainable


def simple_loss(embedding):
  x = constant_op.constant(np.random.rand(2, 3), dtype=dtypes.float32)
  pred = math_ops.matmul(embedding, x)
  loss = pred * pred
  return loss


def extract_keys(test_case, var, slot_vars, status):
  keys, _ = test_case.evaluate(var.export())
  status_keys, _ = test_case.evaluate(status.export())
  slot_keys = []
  for sv in slot_vars:
    sk, _ = test_case.evaluate(sv.export())
    slot_keys.append(sk)
  keys.sort()
  status_keys.sort()
  [sk.sort() for sk in slot_keys]
  return keys, slot_keys, status_keys


def wide_and_deep_model_fn(wide_var, deep_var, embed_dim, ids0, ids1, ids2):
  embed_w0, trainable0 = de.embedding_lookup(wide_var,
                                             ids0,
                                             return_trainable=True)
  embed_w1, trainable1 = de.embedding_lookup(deep_var,
                                             ids1,
                                             return_trainable=True)
  embed_w2, trainable2 = de.embedding_lookup(deep_var,
                                             ids2,
                                             return_trainable=True)

  def layer_fn(entry, dimension, activation=False):
    entry = array_ops.reshape(entry, (-1, dimension, embed_dim))
    dnn_fn = layers.Dense(dimension, use_bias=False)
    batch_normal_fn = layers.BatchNormalization()
    dnn_result = dnn_fn(entry)
    if activation:
      return batch_normal_fn(nn.selu(dnn_result))
    return dnn_result

  wide_branch = layer_fn(embed_w0, 32, activation=True)
  wide_pred = layer_fn(wide_branch, 1)
  wide_logits = math_ops.reduce_mean(wide_pred)

  deep_branch = layer_fn(embed_w2, 16, activation=True)
  deep_pred = layer_fn(deep_branch, 1)
  deep_logits = math_ops.reduce_mean(deep_pred)

  logits = wide_logits + deep_logits
  labels = 0.0
  err_prob = nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
  loss = math_ops.reduce_mean(err_prob)
  return labels, [trainable0, trainable2], loss


default_config = config_pb2.ConfigProto(
    allow_soft_placement=False,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))


class RestrictPolicyV1TestBase(object):

  def common_update_verify(self, optimizer):
    raise NotImplementedError

  def common_restrict_verify(self, optimizer):
    raise NotImplementedError

  # update test
  @test_util.deprecated_graph_mode_only
  def test_adadelta_update(self):
    opt = adadelta.AdadeltaOptimizer()
    self.common_update_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_adagrad_update(self):
    opt = adagrad.AdagradOptimizer(0.1)
    self.common_update_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_adagrad_da_update(self):
    gstep = training_util.create_global_step()
    opt = adagrad_da.AdagradDAOptimizer(0.1, gstep)
    self.common_update_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_adam_update(self):
    opt = adam.AdamOptimizer(0.1)
    self.common_update_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_ftrl_update(self):
    opt = ftrl.FtrlOptimizer(0.1)
    self.common_update_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_gradient_descent_update(self):
    opt = gradient_descent.GradientDescentOptimizer(0.1)
    self.common_update_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_momentum_update(self):
    opt = momentum.MomentumOptimizer(0.1, 0.1)
    self.common_update_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_proximal_adagrad_update(self):
    opt = proximal_adagrad.ProximalAdagradOptimizer(0.1)
    self.common_update_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_pgd_update(self):
    opt = pgd.ProximalGradientDescentOptimizer(0.1)
    self.common_update_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_rmsprop_update(self):
    opt = rmsprop.RMSPropOptimizer(0.1)
    self.common_update_verify(opt)

  # restrict test
  @test_util.deprecated_graph_mode_only
  def test_adadelta_restrict(self):
    opt = adadelta.AdadeltaOptimizer()
    self.common_restrict_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_adagrad_restrict(self):
    opt = adagrad.AdagradOptimizer(0.1)
    self.common_restrict_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_adagrad_da_restrict(self):
    gstep = training_util.create_global_step()
    opt = adagrad_da.AdagradDAOptimizer(0.1, gstep)
    self.common_restrict_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_adam_restrict(self):
    opt = adam.AdamOptimizer(0.1)
    self.common_restrict_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_ftrl_restrict(self):
    opt = ftrl.FtrlOptimizer(0.1)
    self.common_restrict_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_gradient_descent_restrict(self):
    opt = gradient_descent.GradientDescentOptimizer(0.1)
    self.common_restrict_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_momentum_restrict(self):
    opt = momentum.MomentumOptimizer(0.1, 0.1)
    self.common_restrict_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_proximal_adagrad_restrict(self):
    opt = proximal_adagrad.ProximalAdagradOptimizer(0.1)
    self.common_restrict_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_pgd_restrict(self):
    opt = pgd.ProximalGradientDescentOptimizer(0.1)
    self.common_restrict_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_rmsprop_restrict(self):
    opt = rmsprop.RMSPropOptimizer(0.1)
    self.common_restrict_verify(opt)


class TimestampRestrictPolicyV1Test(test.TestCase, RestrictPolicyV1TestBase):

  def common_update_verify(self, optmz):
    with self.session(config=default_config, use_gpu=False):
      ids = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
      var = de.get_variable('sp_var',
                            key_dtype=ids.dtype,
                            value_dtype=dtypes.float32,
                            initializer=-1.,
                            dim=2)
      embed_w, trainable = simple_embedding(var, ids)
      loss = simple_loss(embed_w)

      optmz = de.DynamicEmbeddingOptimizer(optmz)
      train_op = optmz.minimize(loss, var_list=[trainable])
      slot_vars = select_slot_vars([trainable], optmz)

      policy = de.TimestampRestrictPolicy(var, optmz)
      update_op = policy.update_rule()

      self.evaluate(variables.global_variables_initializer())
      self.evaluate([train_op, update_op])

      # Verify variable, slots, and status sizes.
      var_size = self.evaluate(var.size())
      tstp_status_size = self.evaluate(policy.tstp_var.size())
      slot_sizes = self.evaluate([sv.size() for sv in slot_vars])

      self.assertAllEqual(var_size, 3)
      self.assertAllEqual(tstp_status_size, 3)
      self.assertAllEqual(slot_sizes, [3] * len(slot_sizes))

      # Verify keys in var, slots, and status.
      keys, _ = self.evaluate(var.export())
      keys.sort()
      self.assertAllEqual(keys, [1, 2, 3])

      for sv in slot_vars:
        keys, _ = self.evaluate(sv.export())
        keys.sort()
        self.assertAllEqual(keys, [1, 2, 3])

      keys, _ = self.evaluate(policy.tstp_var.export())
      keys.sort()
      self.assertAllEqual(keys, [1, 2, 3])

  def common_restrict_verify(self, optmz):
    first_input = list(range(0, 6))
    second_input = list(range(3, 9))
    all_input = list(range(0, 9))
    remained_keys = [3, 4, 5, 6, 7, 8]
    residue, oversize_residue = 6, 100
    trigger, oversize_trigger = 8, 100

    with self.session(config=default_config, use_gpu=True) as sess:
      ids = array_ops.placeholder(dtypes.int64)
      var = de.get_variable('sp_var',
                            key_dtype=ids.dtype,
                            value_dtype=dtypes.float32,
                            initializer=-1.,
                            dim=2)
      embed_w, trainable = simple_embedding(var, ids)
      loss = simple_loss(embed_w)

      optmz = de.DynamicEmbeddingOptimizer(optmz)
      train_op = optmz.minimize(loss, var_list=[trainable])
      slot_vars = select_slot_vars([trainable], optmz)

      policy = de.TimestampRestrictPolicy(var, optmz)

      update_op = policy.update_rule()
      oversize_restrict_op = policy.restrict_rule(residue=oversize_residue)
      oversize_trigger_restrict_op = policy.restrict_rule(
          residue=residue, trigger=oversize_trigger)
      restrict_op = policy.restrict_rule(residue=residue, trigger=trigger)

      self.evaluate(variables.global_variables_initializer())
      sess.run([train_op, update_op], feed_dict={ids: first_input})
      sess.run([train_op, update_op], feed_dict={ids: second_input})

      # After training, all inputs entered embedding.
      keys, slot_keys, tstp_keys = extract_keys(self, var, slot_vars,
                                                policy.tstp_var)
      self.assertAllEqual(keys, all_input)
      self.assertAllEqual(tstp_keys, all_input)
      [self.assertAllEqual(sk, all_input) for sk in slot_keys]

      # After oversize restriction, nothing happens.
      self.evaluate(oversize_restrict_op)
      keys, slot_keys, tstp_keys = extract_keys(self, var, slot_vars,
                                                policy.tstp_var)
      self.assertAllEqual(keys, all_input)
      self.assertAllEqual(tstp_keys, all_input)
      [self.assertAllEqual(sk, all_input) for sk in slot_keys]

      # After untriggered restriction, only part of keys remain.
      self.evaluate(oversize_trigger_restrict_op)
      keys, slot_keys, tstp_keys = extract_keys(self, var, slot_vars,
                                                policy.tstp_var)
      self.assertAllEqual(keys, all_input)
      self.assertAllEqual(tstp_keys, all_input)
      [self.assertAllEqual(sk, all_input) for sk in slot_keys]

      # After restriction, only part of keys remain.
      self.evaluate(restrict_op)
      keys, slot_keys, tstp_keys = extract_keys(self, var, slot_vars,
                                                policy.tstp_var)
      self.assertAllEqual(keys, remained_keys)
      self.assertAllEqual(tstp_keys, remained_keys)
      [self.assertAllEqual(sk, remained_keys) for sk in slot_keys]


class FrequencyRestrictPolicyV1Test(test.TestCase, RestrictPolicyV1TestBase):

  def common_update_verify(self, optmz):
    with self.session(config=default_config, use_gpu=False):
      ids = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
      var = de.get_variable('sp_var',
                            key_dtype=ids.dtype,
                            value_dtype=dtypes.float32,
                            initializer=-1.,
                            dim=2)
      embed_w, trainable = simple_embedding(var, ids)
      loss = simple_loss(embed_w)

      optmz = de.DynamicEmbeddingOptimizer(optmz)
      train_op = optmz.minimize(loss, var_list=[trainable])
      slot_vars = select_slot_vars([trainable], optmz)

      policy = de.FrequencyRestrictPolicy(var, optmz)
      update_op = policy.update_rule()

      self.evaluate(variables.global_variables_initializer())
      self.evaluate([train_op, update_op])

      # Verify variable, slots, and status sizes.
      var_size = self.evaluate(var.size())
      freq_status_size = self.evaluate(policy.freq_var.size())
      slot_sizes = self.evaluate([sv.size() for sv in slot_vars])

      self.assertAllEqual(var_size, 3)
      self.assertAllEqual(freq_status_size, 3)
      self.assertAllEqual(slot_sizes, [3] * len(slot_sizes))

      # Verify keys in var, slots, and status.
      keys, _ = self.evaluate(var.export())
      keys.sort()
      self.assertAllEqual(keys, [1, 2, 3])

      for sv in slot_vars:
        keys, _ = self.evaluate(sv.export())
        keys.sort()
        self.assertAllEqual(keys, [1, 2, 3])

      keys, _ = self.evaluate(policy.freq_var.export())
      keys.sort()
      self.assertAllEqual(keys, [1, 2, 3])

  def common_restrict_verify(self, optmz):
    first_input = list(range(0, 6))
    second_input = list(range(3, 9))
    all_input = list(range(0, 9))
    remained_keys = [3, 4, 5]
    residue, oversize_residue = 6, 100
    trigger, oversize_trigger = 8, 100

    with self.session(config=default_config, use_gpu=True) as sess:
      ids = array_ops.placeholder(dtypes.int64)
      var = de.get_variable('sp_var',
                            key_dtype=ids.dtype,
                            value_dtype=dtypes.float32,
                            initializer=-1.,
                            dim=2)
      embed_w, trainable = simple_embedding(var, ids)
      loss = simple_loss(embed_w)

      optmz = de.DynamicEmbeddingOptimizer(optmz)
      train_op = optmz.minimize(loss, var_list=[trainable])
      slot_vars = select_slot_vars([trainable], optmz)

      policy = de.FrequencyRestrictPolicy(var, optmz)

      update_op = policy.update_rule()
      oversize_restrict_op = policy.restrict_rule(residue=oversize_residue)
      oversize_trigger_restrict_op = policy.restrict_rule(
          residue=residue, trigger=oversize_trigger)
      restrict_op = policy.restrict_rule(residue=residue, trigger=trigger)

      self.evaluate(variables.global_variables_initializer())
      sess.run([train_op, update_op], feed_dict={ids: first_input})
      sess.run([train_op, update_op], feed_dict={ids: second_input})

      # After training, all inputs entered embedding.
      keys, slot_keys, freq_keys = extract_keys(self, var, slot_vars,
                                                policy.freq_var)
      self.assertAllEqual(keys, all_input)
      self.assertAllEqual(freq_keys, all_input)
      [self.assertAllEqual(sk, all_input) for sk in slot_keys]

      # After oversize restriction, nothing happens.
      self.evaluate(oversize_restrict_op)
      keys, slot_keys, freq_keys = extract_keys(self, var, slot_vars,
                                                policy.freq_var)
      self.assertAllEqual(keys, all_input)
      self.assertAllEqual(freq_keys, all_input)
      [self.assertAllEqual(sk, all_input) for sk in slot_keys]

      # After untriggered restriction, only part of keys remain.
      self.evaluate(oversize_trigger_restrict_op)
      keys, slot_keys, freq_keys = extract_keys(self, var, slot_vars,
                                                policy.freq_var)
      self.assertAllEqual(keys, all_input)
      self.assertAllEqual(freq_keys, all_input)
      [self.assertAllEqual(sk, all_input) for sk in slot_keys]

      # After restriction, only part of keys remain.
      self.evaluate(restrict_op)
      keys, slot_keys, freq_keys = extract_keys(self, var, slot_vars,
                                                policy.freq_var)
      self.assertTrue(all(x in keys for x in remained_keys))
      self.assertAllEqual(len(keys), residue)
      self.assertAllEqual(freq_keys, keys)
      for sk in slot_keys:
        self.assertAllEqual(sk, keys)


class RestrictPolicyV2TestBase(object):

  def common_update_verify_v2(self, optimizer):
    raise NotImplementedError

  def common_restrict_verify_v2(self, optimizer):
    raise NotImplementedError

  # update test
  @test_util.run_in_graph_and_eager_modes
  def test_adadelta_update_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.adadelta.Adadelta(1.0)
    self.common_update_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_adagrad_update_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.adagrad.Adagrad(1.0)
    self.common_update_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_adam_update_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.adam.Adam(1.0)
    self.common_update_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_adamax_update_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.adamax.Adamax(1.0)
    self.common_update_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_ftrl_update_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.ftrl.Ftrl(1.0)
    self.common_update_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_sgd_update_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.gradient_descent.SGD(1.0)
    self.common_update_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_nadam_update_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.nadam.Nadam(1.0)
    self.common_update_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_rmsprop_update_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.rmsprop.RMSprop(1.0)
    self.common_update_verify_v2(opt)

  # restrict test
  @test_util.run_in_graph_and_eager_modes
  def test_adadelta_restrict_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.adadelta.Adadelta(1.0)
    self.common_restrict_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_adagrad_restrict_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.adagrad.Adagrad(1.0)
    self.common_restrict_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_adam_restrict_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.adam.Adam(1.0)
    self.common_restrict_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_adamax_restrict_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.adamax.Adamax(1.0)
    self.common_restrict_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_ftrl_restrict_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.ftrl.Ftrl(1.0)
    self.common_restrict_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_sgd_restrict_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.gradient_descent.SGD(1.0)
    self.common_restrict_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_nadam_restrict_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.nadam.Nadam(1.0)
    self.common_restrict_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_rmsprop_restrict_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.rmsprop.RMSprop(1.0)
    self.common_restrict_verify_v2(opt)


class TimestampRestrictPolicyV2Test(test.TestCase, RestrictPolicyV2TestBase):

  def common_update_verify_v2(self, optmz):
    optmz = de.DynamicEmbeddingOptimizer(optmz)

    with self.session(config=default_config, use_gpu=True):
      ids = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
      trainables, sparse_vars = [], []
      var = de.get_variable('sp_var',
                            key_dtype=ids.dtype,
                            value_dtype=dtypes.float32,
                            initializer=-1.,
                            dim=2)
      policy = de.TimestampRestrictPolicy(var, optmz)

      def loss_fn(var, ids, trainables, sparse_vars):
        embed_w, trainable = simple_embedding(var, ids)
        sparse_vars.append(var)
        trainables.clear()
        trainables.append(trainable)
        return simple_loss(embed_w)

      def var_fn():
        return trainables

      train_op = optmz.minimize(
          lambda: loss_fn(var, ids, trainables, sparse_vars), var_fn)

      slot_vars = select_slot_vars(trainables, optmz)
      update_op = policy.update_rule(trainable_wrappers=trainables)
      self.evaluate(variables.global_variables_initializer())

      for _ in range(2):
        self.evaluate([train_op, update_op])
        self.assertAllEqual(self.evaluate(var.size()), 3)
        self.assertAllEqual(self.evaluate(policy.tstp_var.size()), 3)
        for sv in slot_vars:
          self.assertAllEqual(self.evaluate(sv.size()), 3)

  def common_restrict_verify_v2(self, optmz):
    if not context.executing_eagerly():
      self.skipTest('Skip graph mode test for V2 case.')
    optmz = de.DynamicEmbeddingOptimizer(optmz)
    first_input = list(range(0, 6))
    second_input = list(range(3, 9))
    all_input = list(range(0, 9))
    remained_keys = [3, 4, 5, 6, 7, 8]
    residue, oversize_residue = 6, 100
    trigger, oversize_trigger = 8, 100

    with self.session(config=default_config, use_gpu=True):
      ids_1 = constant_op.constant(first_input, dtype=dtypes.int64)
      ids_2 = constant_op.constant(second_input, dtype=dtypes.int64)
      trainables = []
      var = de.get_variable('sp_var',
                            key_dtype=dtypes.int64,
                            value_dtype=dtypes.float32,
                            initializer=-1.,
                            dim=2)

      def loss_fn(var, ids, trainables):
        embed_w, trainable = simple_embedding(var, ids)
        trainables.clear()
        trainables.append(trainable)
        return simple_loss(embed_w)

      def var_fn():
        return trainables

      policy = de.TimestampRestrictPolicy(var, optmz)

      # train with ids_1
      train_op = optmz.minimize(lambda: loss_fn(var, ids_1, trainables), var_fn)
      self.evaluate(variables.global_variables_initializer())
      update_op = policy.update_rule(trainable_wrappers=trainables)
      slot_vars = select_slot_vars(trainables, optmz)

      self.evaluate([train_op, update_op])
      self.assertAllEqual(self.evaluate(var.size()), 6)
      self.assertAllEqual(self.evaluate(policy.tstp_var.size()), 6)
      for sv in slot_vars:
        self.assertAllEqual(self.evaluate(sv.size()), 6)

      # train with ids_2
      train_op = optmz.minimize(lambda: loss_fn(var, ids_2, trainables), var_fn)
      update_op = policy.update_rule(trainable_wrappers=trainables)

      self.evaluate([train_op, update_op])
      self.assertAllEqual(self.evaluate(var.size()), 9)
      self.assertAllEqual(self.evaluate(policy.tstp_var.size()), 9)
      for sv in slot_vars:
        self.assertAllEqual(self.evaluate(sv.size()), 9)

      # restrict with oversize residue and trigger do not work
      restrict_op = policy.restrict_rule(residue=oversize_residue,
                                         trigger=trigger)
      self.evaluate(restrict_op)

      restrict_op = policy.restrict_rule(residue=residue,
                                         trigger=oversize_trigger)
      self.evaluate(restrict_op)

      self.assertAllEqual(self.evaluate(var.size()), 9)
      self.assertAllEqual(self.evaluate(policy.tstp_var.size()), 9)
      for sv in slot_vars:
        self.assertAllEqual(self.evaluate(sv.size()), 9)

      # restrict with residue and trigger works
      restrict_op = policy.restrict_rule(residue=residue, trigger=trigger)
      self.evaluate(restrict_op)

      keys, slot_keys, tstp_keys = extract_keys(self, var, slot_vars,
                                                policy.tstp_var)
      self.assertAllEqual(keys, remained_keys)
      self.assertAllEqual(tstp_keys, remained_keys)
      [self.assertAllEqual(sk, remained_keys) for sk in slot_keys]


class FrequencyRestrictPolicyV2Test(test.TestCase, RestrictPolicyV2TestBase):

  def common_update_verify_v2(self, optmz):
    optmz = de.DynamicEmbeddingOptimizer(optmz)

    with self.session(config=default_config, use_gpu=True):
      ids = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
      trainables, sparse_vars = [], []
      var = de.get_variable('sp_var',
                            key_dtype=ids.dtype,
                            value_dtype=dtypes.float32,
                            initializer=-1.,
                            dim=2)
      policy = de.FrequencyRestrictPolicy(var, optmz)

      def loss_fn(var, ids, trainables, sparse_vars):
        embed_w, trainable = simple_embedding(var, ids)
        sparse_vars.append(var)
        trainables.clear()
        trainables.append(trainable)
        return simple_loss(embed_w)

      def var_fn():
        return trainables

      train_op = optmz.minimize(
          lambda: loss_fn(var, ids, trainables, sparse_vars), var_fn)

      slot_vars = select_slot_vars(trainables, optmz)
      update_op = policy.update_rule(trainable_wrappers=trainables)
      self.evaluate(variables.global_variables_initializer())

      for _ in range(2):
        self.evaluate([train_op, update_op])
        self.assertAllEqual(self.evaluate(var.size()), 3)
        self.assertAllEqual(self.evaluate(policy.freq_var.size()), 3)
        for sv in slot_vars:
          self.assertAllEqual(self.evaluate(sv.size()), 3)

  def common_restrict_verify_v2(self, optmz):
    if not context.executing_eagerly():
      self.skipTest('Skip graph mode test for V2 case.')
    optmz = de.DynamicEmbeddingOptimizer(optmz)
    first_input = list(range(0, 6))
    second_input = list(range(3, 9))
    all_input = list(range(0, 9))
    remained_keys = [3, 4, 5]
    residue, oversize_residue = 6, 100
    trigger, oversize_trigger = 8, 100

    with self.session(config=default_config, use_gpu=True):
      ids_1 = constant_op.constant(first_input, dtype=dtypes.int64)
      ids_2 = constant_op.constant(second_input, dtype=dtypes.int64)
      trainables = []
      var = de.get_variable('sp_var',
                            key_dtype=dtypes.int64,
                            value_dtype=dtypes.float32,
                            initializer=-1.,
                            dim=2)

      def loss_fn(var, ids, trainables):
        embed_w, trainable = simple_embedding(var, ids)
        trainables.clear()
        trainables.append(trainable)
        return simple_loss(embed_w)

      def var_fn():
        return trainables

      policy = de.FrequencyRestrictPolicy(var, optmz)

      # train with ids_1
      train_op = optmz.minimize(lambda: loss_fn(var, ids_1, trainables), var_fn)
      self.evaluate(variables.global_variables_initializer())
      update_op = policy.update_rule(trainable_wrappers=trainables)
      slot_vars = select_slot_vars(trainables, optmz)

      self.evaluate([train_op, update_op])
      self.assertAllEqual(self.evaluate(var.size()), 6)
      self.assertAllEqual(self.evaluate(policy.freq_var.size()), 6)
      for sv in slot_vars:
        self.assertAllEqual(self.evaluate(sv.size()), 6)

      # train with ids_2
      train_op = optmz.minimize(lambda: loss_fn(var, ids_2, trainables), var_fn)
      update_op = policy.update_rule(trainable_wrappers=trainables)

      self.evaluate([train_op, update_op])
      self.assertAllEqual(self.evaluate(var.size()), 9)
      self.assertAllEqual(self.evaluate(policy.freq_var.size()), 9)
      for sv in slot_vars:
        self.assertAllEqual(self.evaluate(sv.size()), 9)

      # restrict with oversize residue and trigger do not work
      restrict_op = policy.restrict_rule(residue=oversize_residue,
                                         trigger=trigger)
      self.evaluate(restrict_op)

      restrict_op = policy.restrict_rule(residue=residue,
                                         trigger=oversize_trigger)
      self.evaluate(restrict_op)

      self.assertAllEqual(self.evaluate(var.size()), 9)
      self.assertAllEqual(self.evaluate(policy.freq_var.size()), 9)
      for sv in slot_vars:
        self.assertAllEqual(self.evaluate(sv.size()), 9)

      # restrict with residue and trigger works
      restrict_op = policy.restrict_rule(residue=residue, trigger=trigger)
      self.evaluate(restrict_op)

      keys, slot_keys, freq_keys = extract_keys(self, var, slot_vars,
                                                policy.freq_var)
      self.assertTrue(all(x in keys for x in remained_keys))
      self.assertAllEqual(len(keys), residue)
      self.assertAllEqual(freq_keys, keys)
      for sk in slot_keys:
        self.assertAllEqual(sk, keys)


class TrainingWithRestrictorV1Test(test.TestCase):

  def verify_training(self, optmz, policy):
    optmz = de.DynamicEmbeddingOptimizer(optmz)
    residue = 100
    trigger = 150
    embed_dim = 8
    data_len = 32
    max_val = 256

    save_dir = os.path.join(self.get_temp_dir(), 'save_and_restore')
    save_path = os.path.join(tempfile.mktemp(prefix=save_dir), 'restrictor')

    server0 = server_lib.Server.create_local_server()
    server1 = server_lib.Server.create_local_server()
    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server0.target[len('grpc://'):]
    job.tasks[1] = server1.target[len('grpc://'):]

    config = config_pb2.ConfigProto(
        cluster_def=cluster_def,
        experimental=config_pb2.ConfigProto.Experimental(
            share_session_state_in_clusterspec_propagation=True))
    config.allow_soft_placement = False

    def data_fn(shape, max_val):
      return (max_val * np.random.rand(*shape)).astype(np.int64)

    wide_var = de.get_variable('wide',
                               key_dtype=dtypes.int64,
                               value_dtype=dtypes.float32,
                               devices=['/job:worker/task:1'],
                               initializer=-1.,
                               dim=embed_dim)
    deep_var = de.get_variable('deep',
                               key_dtype=dtypes.int64,
                               value_dtype=dtypes.float32,
                               devices=['/job:worker/task:1'],
                               initializer=-1.,
                               dim=embed_dim)
    restrictor = de.VariableRestrictor(var_list=[wide_var, deep_var],
                                       optimizer_list=[optmz],
                                       policy=policy)
    ids0 = array_ops.placeholder(dtypes.int64, name='ids_0')
    ids1 = array_ops.placeholder(dtypes.int64, name='ids_1')
    ids2 = array_ops.placeholder(dtypes.int64, name='ids_2')

    with ops.device('/job:worker/task:0'):
      labels, trainables, loss = wide_and_deep_model_fn(wide_var, deep_var,
                                                        embed_dim, ids0, ids1,
                                                        ids2)

      train_op = optmz.minimize(loss, var_list=trainables)
      update_op = restrictor.update(trainable_wrappers=trainables)
      restrict_op = restrictor.restrict(residue=residue, trigger=trigger)

    with session.Session(server0.target, config=config) as sess:

      # check device placement
      graph = sess.graph
      op_names = [_op.name for _op in graph.get_operations()]
      for name in op_names:
        if 'table' in name.lower():
          op = graph.get_operation_by_name(name)
          self.assertAllEqual(op.device, '/job:worker/task:1')

      sess.run(variables.global_variables_initializer())
      cur_sizes = []
      while all(sz < trigger for sz in cur_sizes):
        cur_sizes = sess.run([wide_var.size(), deep_var.size()])
        feed_dict = {
            ids0: data_fn((data_len, 1), max_val),
            ids1: data_fn((data_len, 1), max_val),
            ids2: data_fn((data_len, 1), max_val),
        }
        sess.run([train_op, update_op], feed_dict=feed_dict)

      # save to checkpoint
      saver = saver_lib.Saver()
      save_return_path = saver.save(sess, save_path)
      self.assertAllEqual(save_return_path, save_path)
      sess.close()

    with session.Session(server0.target, config=config) as sess:
      # restore from checkpoint
      saver.restore(sess, save_path)
      self.assertTrue(all(sz >= trigger for sz in cur_sizes))
      sess.run(restrict_op)
      cur_sizes = sess.run([wide_var.size(), deep_var.size()])
      self.assertTrue(all(sz == residue for sz in cur_sizes))
      slot_vars = select_slot_vars(trainables, optmz)
      for sv in slot_vars:
        self.assertAllEqual(sv.size(), residue)

  @test_util.deprecated_graph_mode_only
  def test_with_timestamp_policy(self):
    optmz = adam.AdamOptimizer(0.1)
    self.verify_training(optmz, de.TimestampRestrictPolicy)

  @test_util.deprecated_graph_mode_only
  def test_with_timestamp_policy(self):
    optmz = adam.AdamOptimizer(0.1)
    self.verify_training(optmz, de.FrequencyRestrictPolicy)


class TrainingWithRestrictorV2Test(test.TestCase):

  def verify_training_v2(self, optmz, policy):
    if not context.executing_eagerly():
      self.skipTest('Test in eager mode only.')

    optmz = de.DynamicEmbeddingOptimizer(optmz)
    residue = 100
    trigger = 150
    embed_dim = 8
    data_len = 32
    max_val = 256

    def data_fn(shape, max_val):
      return (max_val * np.random.rand(*shape)).astype(np.int64)

    with self.session(config=default_config, use_gpu=True) as sess:
      wide_var = de.get_variable('wide',
                                 key_dtype=dtypes.int64,
                                 value_dtype=dtypes.float32,
                                 initializer=-1.,
                                 dim=embed_dim)
      deep_var = de.get_variable('deep',
                                 key_dtype=dtypes.int64,
                                 value_dtype=dtypes.float32,
                                 initializer=-1.,
                                 dim=embed_dim)
      restrictor = de.VariableRestrictor(var_list=[wide_var, deep_var],
                                         optimizer_list=[optmz],
                                         policy=policy)
      trainables = []

      def loss_fn(wide_var, deep_var, trainables):
        ids0, ids1, ids2 = [data_fn((data_len, 1), max_val) for _ in range(3)]
        labels, tws, loss = wide_and_deep_model_fn(wide_var, deep_var,
                                                   embed_dim, ids0, ids1, ids2)
        trainables.clear()
        trainables.extend(tws)
        return loss

      def var_fn():
        return trainables

      cur_sizes = [0, 0]

      while all(sz < trigger for sz in cur_sizes):
        train_op = optmz.minimize(
            lambda: loss_fn(wide_var, deep_var, trainables), var_fn)
        update_op = restrictor.update(trainable_wrappers=trainables)
        self.evaluate([train_op, update_op])
        cur_sizes = self.evaluate([wide_var.size(), deep_var.size()])

      self.assertTrue(all(sz >= trigger for sz in cur_sizes))

      restrict_op = restrictor.restrict(residue=residue, trigger=trigger)
      self.evaluate(restrict_op)

      cur_sizes = self.evaluate([wide_var.size(), deep_var.size()])
      self.assertTrue(all(sz == residue for sz in cur_sizes))
      slot_vars = select_slot_vars(trainables, optmz)
      for sv in slot_vars:
        self.assertAllEqual(sv.size(), residue)

  @test_util.run_in_graph_and_eager_modes
  def test_with_timestamp_policy_v2(self):
    optmz = optimizer_v2.adam.Adam(1.0)
    self.verify_training_v2(optmz, de.TimestampRestrictPolicy)

  @test_util.run_in_graph_and_eager_modes
  def test_with_timestamp_policy_v2(self):
    optmz = optimizer_v2.adam.Adam(1.0)
    self.verify_training_v2(optmz, de.FrequencyRestrictPolicy)


if __name__ == '__main__':
  test.main()
