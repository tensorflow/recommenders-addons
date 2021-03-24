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
"""Unit tests of restrict policies"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tempfile
import time

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.keras import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
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
from tensorflow.python.training import server_lib
from tensorflow.python.training import training_util


def _simple_loss(embedding):
  x = constant_op.constant(np.random.rand(2, 3), dtype=dtypes.float32)
  pred = math_ops.matmul(embedding, x)
  loss = pred * pred
  return loss


default_config = config_pb2.ConfigProto(
    allow_soft_placement=False,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))


class RestrictPolicyV1TestBase(object):

  def commonly_apply_update_verify(self):
    raise NotImplementedError

  def commonly_apply_restriction_verify(self, optimizer):
    raise NotImplementedError

  # apply update test
  @test_util.deprecated_graph_mode_only
  def test_apply_update(self):
    self.commonly_apply_update_verify()

  # apply restrict test
  @test_util.deprecated_graph_mode_only
  def test_adadelta_apply_restriction(self):
    opt = adadelta.AdadeltaOptimizer()
    self.commonly_apply_restriction_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_adagrad_apply_restriction(self):
    opt = adagrad.AdagradOptimizer(0.1)
    self.commonly_apply_restriction_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_adagrad_da_apply_restriction(self):
    gstep = training_util.create_global_step()
    opt = adagrad_da.AdagradDAOptimizer(0.1, gstep)
    self.commonly_apply_restriction_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_adam_apply_restriction(self):
    opt = adam.AdamOptimizer(0.1)
    self.commonly_apply_restriction_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_ftrl_apply_restriction(self):
    opt = ftrl.FtrlOptimizer(0.1)
    self.commonly_apply_restriction_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_gradient_descent_apply_restriction(self):
    opt = gradient_descent.GradientDescentOptimizer(0.1)
    self.commonly_apply_restriction_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_momentum_apply_restriction(self):
    opt = momentum.MomentumOptimizer(0.1, 0.1)
    self.commonly_apply_restriction_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_proximal_adagrad_apply_restriction(self):
    opt = proximal_adagrad.ProximalAdagradOptimizer(0.1)
    self.commonly_apply_restriction_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_pgd_apply_restriction(self):
    opt = pgd.ProximalGradientDescentOptimizer(0.1)
    self.commonly_apply_restriction_verify(opt)

  @test_util.deprecated_graph_mode_only
  def test_rmsprop_apply_restriction(self):
    opt = rmsprop.RMSPropOptimizer(0.1)
    self.commonly_apply_restriction_verify(opt)


class TimestampRestrictPolicyV1Test(test.TestCase, RestrictPolicyV1TestBase):

  def commonly_apply_update_verify(self):
    first_inputs = np.array(range(3), dtype=np.int64)
    second_inputs = np.array(range(1, 4), dtype=np.int64)
    overdue_features = np.array([0], dtype=np.int64)
    updated_features = np.array(range(1, 4), dtype=np.int64)
    with session.Session(config=default_config) as sess:
      ids = array_ops.placeholder(dtypes.int64)
      var = de.get_variable('sp_var',
                            key_dtype=ids.dtype,
                            value_dtype=dtypes.float32,
                            initializer=-0.1,
                            init_size=256,
                            dim=2)
      embed_w, trainable = de.embedding_lookup(var, ids, return_trainable=True)
      policy = de.TimestampRestrictPolicy(var)
      update_op = policy.apply_update(ids)

      self.assertAllEqual(sess.run(policy.status.size()), 0)
      sess.run(update_op, feed_dict={ids: first_inputs})
      self.assertAllEqual(sess.run(policy.status.size()), 3)
      time.sleep(1)
      sess.run(update_op, feed_dict={ids: second_inputs})
      self.assertAllEqual(sess.run(policy.status.size()), 4)

      keys, tstp = sess.run(policy.status.export())
      kvs = sorted(dict(zip(keys, tstp)).items())
      tstp = np.array([x[1] for x in kvs])
      for x in tstp[overdue_features]:
        for y in tstp[updated_features]:
          self.assertLess(x, y)

  def commonly_apply_restriction_verify(self, optimizer):
    first_inputs = np.array(range(6), dtype=np.int64)
    second_inputs = np.array(range(4, 9), dtype=np.int64)
    overdue_features = np.array(range(4), dtype=np.int64)
    updated_features = np.array(range(4, 9), dtype=np.int64)
    all_input_features = np.array(range(9), dtype=np.int64)
    embedding_dim = 2
    oversize_trigger = 100
    optimizer = de.DynamicEmbeddingOptimizer(optimizer)

    with session.Session(config=default_config) as sess:
      ids = array_ops.placeholder(dtypes.int64)
      var = de.get_variable('sp_var',
                            key_dtype=ids.dtype,
                            value_dtype=dtypes.float32,
                            initializer=-0.1,
                            dim=embedding_dim,
                            restrict_policy=de.TimestampRestrictPolicy)
      embed_w, trainable = de.embedding_lookup(var, ids, return_trainable=True)
      loss = _simple_loss(embed_w)
      train_op = optimizer.minimize(loss, var_list=[trainable])

      slot_params = [
          optimizer.get_slot(trainable, name).params
          for name in optimizer.get_slot_names()
      ]
      all_vars = [var] + slot_params + [var.restrict_policy.status]

      sess.run(variables.global_variables_initializer())

      sess.run([train_op], feed_dict={ids: first_inputs})
      time.sleep(1)
      sess.run([train_op], feed_dict={ids: second_inputs})
      for v in all_vars:
        self.assertAllEqual(sess.run(v.size()), 9)
      keys, tstp = sess.run(var.restrict_policy.status.export())
      kvs = sorted(dict(zip(keys, tstp)).items())
      tstp = np.array([x[1] for x in kvs])
      for x in tstp[overdue_features]:
        for y in tstp[updated_features]:
          self.assertLess(x, y)

      sess.run(
          var.restrict_policy.apply_restriction(len(updated_features),
                                                trigger=oversize_trigger))
      for v in all_vars:
        self.assertAllEqual(sess.run(v.size()), len(all_input_features))

      sess.run(
          var.restrict_policy.apply_restriction(len(updated_features),
                                                trigger=len(updated_features)))
      for v in all_vars:
        self.assertAllEqual(sess.run(v.size()), len(updated_features))
      keys, _ = sess.run(var.export())
      self.assertAllEqual(keys, updated_features)


class FrequencyRestrictPolicyV1Test(test.TestCase, RestrictPolicyV1TestBase):

  def commonly_apply_update_verify(self):
    first_inputs = np.array(range(3), dtype=np.int64)
    second_inputs = np.array(range(1, 4), dtype=np.int64)
    overdue_features = np.array([0, 3], dtype=np.int64)
    updated_features = np.array(range(1, 3), dtype=np.int64)
    with session.Session(config=default_config) as sess:
      ids = array_ops.placeholder(dtypes.int64)
      var = de.get_variable('sp_var',
                            key_dtype=ids.dtype,
                            value_dtype=dtypes.float32,
                            initializer=-0.1,
                            dim=2)
      embed_w, trainable = de.embedding_lookup(var, ids, return_trainable=True)
      policy = de.FrequencyRestrictPolicy(var)
      update_op = policy.apply_update(ids)

      self.assertAllEqual(sess.run(policy.status.size()), 0)
      sess.run(update_op, feed_dict={ids: first_inputs})
      self.assertAllEqual(sess.run(policy.status.size()), 3)
      time.sleep(1)
      sess.run(update_op, feed_dict={ids: second_inputs})
      self.assertAllEqual(sess.run(policy.status.size()), 4)

      keys, freq = sess.run(policy.status.export())
      kvs = sorted(dict(zip(keys, freq)).items())
      freq = np.array([x[1] for x in kvs])
      for x in freq[overdue_features]:
        for y in freq[updated_features]:
          self.assertLess(x, y)

  def commonly_apply_restriction_verify(self, optimizer):
    first_inputs = np.array(range(6), dtype=np.int64)
    second_inputs = np.array(range(4, 9), dtype=np.int64)
    overdue_features = np.array([0, 1, 2, 3, 6, 7, 8], dtype=np.int64)
    updated_features = np.array(range(4, 6), dtype=np.int64)
    all_input_features = np.array(range(9), dtype=np.int64)
    embedding_dim = 2
    oversize_trigger = 100
    optimizer = de.DynamicEmbeddingOptimizer(optimizer)

    with session.Session(config=default_config) as sess:
      ids = array_ops.placeholder(dtypes.int64)
      var = de.get_variable('sp_var',
                            key_dtype=ids.dtype,
                            value_dtype=dtypes.float32,
                            initializer=-0.1,
                            dim=embedding_dim,
                            restrict_policy=de.FrequencyRestrictPolicy)
      embed_w, trainable = de.embedding_lookup(var, ids, return_trainable=True)
      loss = _simple_loss(embed_w)
      train_op = optimizer.minimize(loss, var_list=[trainable])

      slot_params = [
          optimizer.get_slot(trainable, name).params
          for name in optimizer.get_slot_names()
      ]
      all_vars = [var] + slot_params + [var.restrict_policy.status]

      sess.run(variables.global_variables_initializer())

      sess.run([train_op], feed_dict={ids: first_inputs})
      sess.run([train_op], feed_dict={ids: second_inputs})
      for v in all_vars:
        self.assertAllEqual(sess.run(v.size()), 9)
      keys, freq = sess.run(var.restrict_policy.status.export())
      kvs = sorted(dict(zip(keys, freq)).items())
      freq = np.array([x[1] for x in kvs])
      for x in freq[overdue_features]:
        for y in freq[updated_features]:
          self.assertLess(x, y)

      sess.run(
          var.restrict_policy.apply_restriction(len(updated_features),
                                                trigger=oversize_trigger))
      for v in all_vars:
        self.assertAllEqual(sess.run(v.size()), len(all_input_features))

      sess.run(
          var.restrict_policy.apply_restriction(len(updated_features),
                                                trigger=len(updated_features)))
      for v in all_vars:
        self.assertAllEqual(sess.run(v.size()), len(updated_features))
      keys, _ = sess.run(var.export())
      self.assertAllEqual(keys, updated_features)


class RestrictPolicyV2TestBase(object):

  def commonly_apply_update_verify_v2(self):
    raise NotImplementedError

  def commonly_apply_restriction_verify_v2(self, optimizer):
    raise NotImplementedError

  # track test
  @test_util.run_in_graph_and_eager_modes
  def test_apply_update_verify_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    self.commonly_apply_update_verify_v2()

  # apply restrict test
  @test_util.run_in_graph_and_eager_modes
  def test_adadelta_apply_restriction_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.adadelta.Adadelta(1.0)
    self.commonly_apply_restriction_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_adagrad_apply_restriction_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.adagrad.Adagrad(1.0)
    self.commonly_apply_restriction_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_adam_apply_restriction_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.adam.Adam(1.0)
    self.commonly_apply_restriction_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_adamax_apply_restriction_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.adamax.Adamax(1.0)
    self.commonly_apply_restriction_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_ftrl_apply_restriction_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.ftrl.Ftrl(1.0)
    self.commonly_apply_restriction_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_sgd_apply_restriction_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.gradient_descent.SGD(1.0)
    self.commonly_apply_restriction_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_nadam_apply_restriction_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.nadam.Nadam(1.0)
    self.commonly_apply_restriction_verify_v2(opt)

  @test_util.run_in_graph_and_eager_modes
  def test_rmsprop_apply_restriction_v2(self):
    if test_util.is_gpu_available():
      self.skipTest('Skip GPU test for no GPU kernel')
    opt = optimizer_v2.rmsprop.RMSprop(1.0)
    self.commonly_apply_restriction_verify_v2(opt)


class TimestampRestrictPolicyV2Test(test.TestCase, RestrictPolicyV2TestBase):

  def commonly_apply_update_verify_v2(self):
    if not context.executing_eagerly():
      self.skipTest('Skip graph mode test.')

    first_inputs = np.array(range(6), dtype=np.int64)
    second_inputs = np.array(range(3, 9), dtype=np.int64)
    overdue_features = np.array(range(3), dtype=np.int64)
    updated_features = np.array(range(3, 9), dtype=np.int64)
    all_features = np.array(range(9), dtype=np.int64)

    with self.session(config=default_config):
      var = de.get_variable('sp_var',
                            key_dtype=dtypes.int64,
                            value_dtype=dtypes.float32,
                            initializer=-0.1,
                            dim=2)
      embed_w, trainable = de.embedding_lookup(var,
                                               first_inputs,
                                               return_trainable=True)
      policy = de.TimestampRestrictPolicy(var)

      self.assertAllEqual(policy.status.size(), 0)
      policy.apply_update(first_inputs)
      self.assertAllEqual(policy.status.size(), len(first_inputs))
      time.sleep(1)
      policy.apply_update(second_inputs)
      self.assertAllEqual(policy.status.size(), len(all_features))

      keys, tstp = policy.status.export()
      kvs = sorted(dict(zip(keys.numpy(), tstp.numpy())).items())
      tstp = np.array([x[1] for x in kvs])
      for x in tstp[overdue_features]:
        for y in tstp[updated_features]:
          self.assertLess(x, y)

  def commonly_apply_restriction_verify_v2(self, optimizer):
    if not context.executing_eagerly():
      self.skipTest('Skip graph mode test.')

    optimizer = de.DynamicEmbeddingOptimizer(optimizer)
    first_inputs = np.array(range(6), dtype=np.int64)
    second_inputs = np.array(range(3, 9), dtype=np.int64)
    overdue_features = np.array(range(3), dtype=np.int64)
    updated_features = np.array(range(3, 9), dtype=np.int64)
    all_inputs = np.array(range(9), dtype=np.int64)
    oversize_trigger = 100
    trainables = []

    with self.session(config=default_config):
      var = de.get_variable('sp_var',
                            key_dtype=dtypes.int64,
                            value_dtype=dtypes.float32,
                            initializer=-0.1,
                            dim=2,
                            restrict_policy=de.TimestampRestrictPolicy)

      def loss_fn(var, features, trainables):
        embed_w, trainable = de.embedding_lookup(var,
                                                 features,
                                                 return_trainable=True)
        trainables.clear()
        trainables.append(trainable)
        return _simple_loss(embed_w)

      def var_fn():
        return trainables

      optimizer.minimize(lambda: loss_fn(var, first_inputs, trainables), var_fn)
      self.assertAllEqual(var.size(), len(first_inputs))
      slot_params = [
          optimizer.get_slot(trainables[0], name).params
          for name in optimizer.get_slot_names()
      ]
      all_vars = [var] + slot_params + [var.restrict_policy.status]

      time.sleep(1)
      optimizer.minimize(lambda: loss_fn(var, second_inputs, trainables),
                         var_fn)
      for v in all_vars:
        keys, _ = v.export()
        self.assertAllEqual(sorted(keys), all_inputs)
      keys, tstp = var.restrict_policy.status.export()
      kvs = sorted(dict(zip(keys.numpy(), tstp.numpy())).items())
      tstp = np.array([x[1] for x in kvs])
      for x in tstp[overdue_features]:
        for y in tstp[updated_features]:
          self.assertLess(x, y)

      var.restrict_policy.apply_restriction(len(updated_features),
                                            trigger=oversize_trigger)
      for v in all_vars:
        self.assertAllEqual(v.size(), len(all_inputs))
      var.restrict_policy.apply_restriction(len(updated_features),
                                            trigger=len(updated_features))
      for v in all_vars:
        keys, _ = v.export()
        self.assertAllEqual(sorted(keys), updated_features)


class FrequencyRestrictPolicyV2Test(test.TestCase, RestrictPolicyV2TestBase):

  def commonly_apply_update_verify_v2(self):
    if not context.executing_eagerly():
      self.skipTest('Skip graph mode test.')

    first_inputs = np.array(range(6), dtype=np.int64)
    second_inputs = np.array(range(3, 9), dtype=np.int64)
    overdue_features = np.array([0, 1, 2, 6, 7, 8], dtype=np.int64)
    updated_features = np.array(range(3, 6), dtype=np.int64)
    all_features = np.array(range(9), dtype=np.int64)

    with self.session(config=default_config):
      var = de.get_variable('sp_var',
                            key_dtype=dtypes.int64,
                            value_dtype=dtypes.float32,
                            initializer=-0.1,
                            dim=2)
      embed_w, trainable = de.embedding_lookup(var,
                                               first_inputs,
                                               return_trainable=True)
      policy = de.FrequencyRestrictPolicy(var)

      self.assertAllEqual(policy.status.size(), 0)
      policy.apply_update(first_inputs)
      self.assertAllEqual(policy.status.size(), len(first_inputs))
      time.sleep(1)
      policy.apply_update(second_inputs)
      self.assertAllEqual(policy.status.size(), len(all_features))

      keys, freq = policy.status.export()
      kvs = sorted(dict(zip(keys.numpy(), freq.numpy())).items())
      freq = np.array([x[1] for x in kvs])
      for x in freq[overdue_features]:
        for y in freq[updated_features]:
          self.assertLess(x, y)

  def commonly_apply_restriction_verify_v2(self, optimizer):
    if not context.executing_eagerly():
      self.skipTest('Skip graph mode test.')

    optimizer = de.DynamicEmbeddingOptimizer(optimizer)
    first_inputs = np.array(range(6), dtype=np.int64)
    second_inputs = np.array(range(3, 9), dtype=np.int64)
    overdue_features = np.array([0, 1, 2, 6, 7, 8], dtype=np.int64)
    updated_features = np.array(range(3, 6), dtype=np.int64)
    all_inputs = np.array(range(9), dtype=np.int64)
    oversize_trigger = 100
    trainables = []

    with self.session(config=default_config):
      var = de.get_variable('sp_var',
                            key_dtype=dtypes.int64,
                            value_dtype=dtypes.float32,
                            initializer=-0.1,
                            dim=2,
                            restrict_policy=de.FrequencyRestrictPolicy)

      def loss_fn(var, features, trainables):
        embed_w, trainable = de.embedding_lookup(var,
                                                 features,
                                                 return_trainable=True)
        trainables.clear()
        trainables.append(trainable)
        return _simple_loss(embed_w)

      def var_fn():
        return trainables

      optimizer.minimize(lambda: loss_fn(var, first_inputs, trainables), var_fn)
      self.assertAllEqual(var.size(), len(first_inputs))
      slot_params = [
          optimizer.get_slot(trainables[0], name).params
          for name in optimizer.get_slot_names()
      ]
      all_vars = [var] + slot_params + [var.restrict_policy.status]

      optimizer.minimize(lambda: loss_fn(var, second_inputs, trainables),
                         var_fn)
      for v in all_vars:
        keys, _ = v.export()
        self.assertAllEqual(sorted(keys), all_inputs)
      keys, freq = var.restrict_policy.status.export()
      kvs = sorted(dict(zip(keys.numpy(), freq.numpy())).items())
      freq = np.array([x[1] for x in kvs])
      for x in freq[overdue_features]:
        for y in freq[updated_features]:
          self.assertLess(x, y)

      var.restrict_policy.apply_restriction(len(updated_features),
                                            trigger=oversize_trigger)
      for v in all_vars:
        self.assertAllEqual(v.size(), len(all_inputs))
      var.restrict_policy.apply_restriction(len(updated_features),
                                            trigger=len(updated_features))
      for v in all_vars:
        keys, _ = v.export()
        self.assertAllEqual(sorted(keys), updated_features)


if __name__ == '__main__':
  test.main()
