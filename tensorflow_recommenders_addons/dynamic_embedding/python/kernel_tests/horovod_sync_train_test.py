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
"""unit tests of dynamic embedding optimizer ops
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import pytest
import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import monitored_session
from tensorflow.python.training import training_util

default_config = config_pb2.ConfigProto(
    allow_soft_placement=True,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))


class HorovodTest(test.TestCase):

  @test_util.deprecated_graph_mode_only
  def test_adam_minimize_trainable(self):
    base_opt = adam.AdamOptimizer(1.0)
    test_opt = adam.AdamOptimizer(1.0)
    self.common_minimize_trainable(base_opt, test_opt, name="adam")

  def common_minimize_trainable(self, base_opt, test_opt, name):
    tf.config.set_soft_device_placement(True)
    hvd.init()
    base_opt = de.DynamicEmbeddingOptimizer(base_opt, synchronous=True)
    for dtype, run_step, dim in itertools.product([dtypes.float32], [1], [10]):
      x = tf.random.uniform(shape=[32, dim])
      y = tf.zeros([32, 1])

      global_step = training_util.create_global_step()

      base_weight = tf.compat.v1.get_variable(name="base_weights",
                                              initializer=tf.ones([10, 1]))

      base_logits = tf.nn.relu(math_ops.matmul(x, base_weight))
      base_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
                                                          logits=base_logits)

      base_opt_op = base_opt.minimize(base_loss,
                                      global_step,
                                      var_list=[base_weight])

      test_weight = tf.compat.v1.get_variable(name="test_weights",
                                              initializer=tf.ones([10, 1]))

      test_logits = tf.nn.relu(math_ops.matmul(x, test_weight))
      test_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
                                                          logits=test_logits)

      grads_and_vars = test_opt.compute_gradients(test_loss,
                                                  var_list=[test_weight])
      var_list = []
      aggregated_grad = []
      for grad, var in grads_and_vars:
        var_list.append(var)
        aggregated_grad.append(hvd.allreduce(grad, op=hvd.Sum))
      aggregated_grads_and_vars = zip(aggregated_grad, var_list)
      test_opt_op = test_opt.apply_gradients(aggregated_grads_and_vars,
                                             global_step)

      with monitored_session.MonitoredTrainingSession(
          is_chief=True, config=default_config) as sess:

        for _ in range(run_step):
          sess.run(base_opt_op)
          sess.run(test_opt_op)

        self.assertAllCloseAccordingToType(
            sess.run(base_weight),
            sess.run(test_weight),
            msg="Cond:{},{},{}".format(dtype, run_step, dim),
        )


if __name__ == "__main__":
  test.main()
