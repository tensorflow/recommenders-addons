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
# Copyright (c) 2017, Alibaba Inc.
# All right reserved.
#
# Author: Chen Ding <cnady.dc@alibaba-inc.com>
# Created: 2018/03/26
# Description:
# ==============================================================================
"""Tests for tensorflow_recommenders_addons.python.ops.embedding_variable."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.training import saver as saver_module
from tensorflow.python.training import training_util
from tensorflow.python.ops import variables
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import loader

from tensorflow_recommenders_addons.embedding_variable.python.ops import embedding_variable_ops
from tensorflow_recommenders_addons.embedding_variable.python.optimizers import gradient_descent


@test_util.run_all_in_graph_and_eager_modes
class EmbeddingVariableTest(test.TestCase):

  def testEmbeddingVariableForTypeNotMatch(self):
    with self.assertRaises(errors.InvalidArgumentError):
      ev = embedding_variable_ops.EmbeddingVariable(
          embedding_dim=3,
          ktype=dtypes.int32,
          initializer=init_ops.ones_initializer(dtypes.float32))
      emb = embedding_ops.embedding_lookup(
          ev, math_ops.cast([0, 1, 2, 5, 6, 7], dtypes.int64))

  def testEmbeddingVariableForGetShape(self):
    ev = embedding_variable_ops.EmbeddingVariable(
        embedding_dim=3, initializer=init_ops.ones_initializer(dtypes.float32))
    emb = embedding_ops.embedding_lookup(
        ev, math_ops.cast([0, 1, 2, 5, 6, 7], dtypes.int32))
    shape = ev.total_count()
    init = variables.global_variables_initializer()
    self.assertEqual(None, self.evaluate(init))
    self.evaluate(emb)
    self.assertAllEqual([6, 3], self.evaluate(shape))

  def testEmbeddingVariableForGeneralConstInitializer(self):
    ev = embedding_variable_ops.EmbeddingVariable(
        embedding_dim=3,
        ktype=dtypes.int64,
        initializer=init_ops.ones_initializer(dtypes.float32))
    emb = embedding_ops.embedding_lookup(ev, math_ops.cast([1, 6],
                                                           dtypes.int64))
    init = variables.global_variables_initializer()
    self.assertEqual(None, self.evaluate(init))
    self.assertAllEqual([[1., 1., 1.]] * 2, self.evaluate(emb))

  def testEmbeddingVariableForGradientDescent(self):
    ev = embedding_variable_ops.EmbeddingVariable(
        embedding_dim=3,
        ktype=dtypes.int64,
        initializer=init_ops.ones_initializer(dtypes.float32))

    def loss_fn(ev):
      emb = embedding_ops.embedding_lookup(
          ev, math_ops.cast([0, 1, 2, 5, 6, 7], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      return loss

    gs = training_util.get_or_create_global_step()
    opt = gradient_descent.GradientDescentOptimizer(0.1)
    g_v = opt.compute_gradients(lambda: loss_fn(ev), [ev])
    train_op = opt.apply_gradients(g_v)
    emb = embedding_ops.embedding_lookup(
        ev, math_ops.cast([0, 1, 2, 5, 6, 7], dtypes.int64))
    init = variables.global_variables_initializer()
    self.assertEqual(None, self.evaluate(init))
    self.assertEqual(None, self.evaluate(train_op))
    emb_result = self.evaluate(emb)
    grad_result = self.evaluate(g_v[0][0])
    for i in range(6):
      for j in range(3):
        self.assertAlmostEqual(.8, emb_result[i][j], delta=1e-05)
        self.assertAlmostEqual(2., grad_result.values[i][j], delta=1e-05)

  @test_util.deprecated_graph_mode_only
  def testEmbeddingVariableForSaveRestore(self):
    ev = embedding_variable_ops.EmbeddingVariable(
        embedding_dim=2,
        initializer=init_ops.random_normal_initializer(),
        ktype=dtypes.int64)
    var_emb = embedding_ops.embedding_lookup(
        ev, math_ops.cast([0, 1, 2], dtypes.int64))
    loss = math_ops.reduce_sum(var_emb)
    optimizer = gradient_descent.GradientDescentOptimizer(0.1)
    with ops.control_dependencies([var_emb]):
      opt = optimizer.minimize(loss)
    saver = saver_module.Saver()
    init = variables.global_variables_initializer()
    with session.Session() as sess:
      sess.run([init])
      sess.run([opt, var_emb])
      sess.run([opt, var_emb])
      sess.run([opt, var_emb])
      save = sess.run(var_emb)
      saver.save(sess, "ckpt")
    with session.Session() as sess:
      saver.restore(sess, "ckpt")
      restore = sess.run(var_emb)
    for i in range(3):
      for j in range(2):
        self.assertAlmostEqual(save[i][j], restore[i][j], delta=1e-05)


if __name__ == "__main__":
  test.main()
