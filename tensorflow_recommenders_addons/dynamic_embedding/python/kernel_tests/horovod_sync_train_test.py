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
import os
import tensorflow as tf

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import monitored_session
from tensorflow.python.training.optimizer import Optimizer as tf1_opt
from tensorflow.python.training import training_util
from tensorflow_recommenders_addons.utils.check_platform import is_macos, is_arm64

default_config = config_pb2.ConfigProto(
    allow_soft_placement=True,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))


def get_emb_sequential_model(emb_t, opt, *args, **kwargs):
  l0 = tf.keras.layers.InputLayer(input_shape=(None,), dtype=dtypes.int64)
  l1 = emb_t(*args, **kwargs)
  l2 = tf.keras.layers.Dense(8, 'relu', kernel_initializer='zeros')
  l3 = tf.keras.layers.Dense(1, 'sigmoid', kernel_initializer='zeros')
  if emb_t == tf.keras.layers.Embedding:
    model = tf.keras.Sequential([l0, l1, l2, l3])
  elif emb_t == de.keras.layers.HvdAllToAllEmbedding:
    model = tf.keras.Sequential([l0, l1, l2, l3])
  else:
    raise TypeError('Unsupported embedding layer {}'.format(emb_t))
  model.compile(optimizer=opt, loss='mean_absolute_error')
  return model


class HorovodTest(test.TestCase):

  @test_util.deprecated_graph_mode_only
  def test_adam_minimize_trainable(self):
    if (is_macos() and is_arm64()):
      self.skipTest(
          "Apple silicon devices don't support synchronous training based on Horovod."
      )

    base_opt = adam.AdamOptimizer(1.0)
    test_opt = adam.AdamOptimizer(1.0)
    self.common_minimize_trainable_v1(base_opt, test_opt, name="adam")
    tf.keras.backend.clear_session()
    keras_base_opt = tf.keras.optimizers.Adam(1.0)
    keras_test_opt = tf.keras.optimizers.Adam(1.0)
    self.common_minimize_trainable_v2(keras_base_opt,
                                      keras_test_opt,
                                      name="keras_adam")

  @test_util.run_all_in_graph_and_eager_modes
  def test_all_to_all_embedding_trainable(self):
    if (is_macos() and is_arm64()):
      self.skipTest(
          "Apple silicon devices don't support synchronous training based on Horovod."
      )
    keras_base_opt = tf.keras.optimizers.Adam(1.0)
    keras_test_opt = tf.keras.optimizers.Adam(1.0)
    self.common_all_to_all_embedding_trainable_v2(keras_base_opt,
                                                  keras_test_opt,
                                                  name="keras_adam")

  def common_minimize_trainable_v1(self, base_opt, test_opt, name):
    # TODO(rhdong): Recover the testing, if the horovod import error is fixed on macOS+TF2.7+.
    try:
      import horovod.tensorflow as hvd
    except (NotFoundError):
      self.skipTest(
          "Skip the test for horovod import error with Tensorflow-2.7.0 on MacOS-12."
      )

    tf.config.set_soft_device_placement(True)

    hvd.init()

    # These cases need 2 GPUs at least if available.
    logical_devices = tf.config.list_logical_devices('GPU')
    _device = "GPU" if len(logical_devices) >= hvd.size() else "CPU"
    _device_id = hvd.local_rank(
    ) if _device == "GPU" and len(logical_devices) >= 2 else 0

    if _device == "GPU":
      os.environ["CUDA_VISIBLE_DEVICES"] = str(_device_id)

    base_opt = de.DynamicEmbeddingOptimizer(base_opt, synchronous=True)
    for dtype, run_step, dim in itertools.product([dtypes.float32], [1], [10]):
      print("device=", "/{}:{}".format(_device, _device_id))
      with tf.device("/{}:{}".format(_device, _device_id)):
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

  def common_minimize_trainable_v2(self, base_opt, test_opt, name):
    # TODO(rhdong): Recover the testing, if the horovod import error is fixed on macOS+TF2.7+.
    try:
      import horovod.tensorflow as hvd
    except (NotFoundError):
      self.skipTest(
          "Skip the test for horovod import error with Tensorflow-2.7.0 on MacOS-12."
      )

    tf.config.set_soft_device_placement(True)

    hvd.init()

    # These cases need 2 GPUs at least if available.
    logical_devices = tf.config.list_logical_devices('GPU')
    _device = "GPU" if len(logical_devices) >= hvd.size() else "CPU"
    _device_id = hvd.local_rank(
    ) if _device == "GPU" and len(logical_devices) >= 2 else 0

    if _device == "GPU":
      os.environ["CUDA_VISIBLE_DEVICES"] = str(_device_id)

    base_opt = de.DynamicEmbeddingOptimizer(base_opt, synchronous=True)
    for dtype, run_step, dim in itertools.product([dtypes.float32], [1], [10]):
      print("device=", "/{}:{}".format(_device, _device_id))
      with tf.device("/{}:{}".format(_device, _device_id)):
        x = tf.random.uniform(shape=[32, dim])
        y = tf.zeros([32, 1])

        with tf.GradientTape() as tape_base:
          base_weight = tf.compat.v1.get_variable(name="base_weights",
                                                  initializer=tf.ones([10, 1]))

          base_logits = tf.nn.relu(math_ops.matmul(x, base_weight))
          base_loss = tf.nn.sigmoid_cross_entropy_with_logits(
              labels=y, logits=base_logits)

          # grad_base = tape_base.gradient(base_loss, base_weight)
          # base_opt.
        base_opt_op = base_opt.minimize(base_loss,
                                        var_list=[base_weight],
                                        tape=tape_base)

        with tf.GradientTape() as tape_test:
          test_weight = tf.compat.v1.get_variable(name="test_weights",
                                                  initializer=tf.ones([10, 1]))

          test_logits = tf.nn.relu(math_ops.matmul(x, test_weight))
          test_loss = tf.nn.sigmoid_cross_entropy_with_logits(
              labels=y, logits=test_logits)

        grads_and_vars = test_opt._compute_gradients(test_loss,
                                                     var_list=[test_weight],
                                                     tape=tape_test)
        var_list = []
        aggregated_grad = []
        for grad, var in grads_and_vars:
          var_list.append(var)
          aggregated_grad.append(hvd.allreduce(grad, op=hvd.Sum))
        aggregated_grads_and_vars = zip(aggregated_grad, var_list)
        test_opt_op = test_opt.apply_gradients(aggregated_grads_and_vars)

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

  def common_all_to_all_embedding_trainable_v2(self, base_opt, test_opt, name):
    # TODO(rhdong): Recover the testing, if the horovod import error is fixed on macOS+TF2.7+.
    try:
      import horovod.tensorflow as hvd
    except (NotFoundError):
      self.skipTest(
          "Skip the test for horovod import error with Tensorflow-2.7.0 on MacOS-12."
      )

    tf.config.set_soft_device_placement(True)

    hvd.init()

    # These cases need 2 GPUs at least if available.
    logical_devices = tf.config.list_logical_devices('GPU')
    _device = "GPU" if len(logical_devices) >= hvd.size() else "CPU"
    _device_id = hvd.local_rank(
    ) if _device == "GPU" and len(logical_devices) >= 2 else 0

    if _device == "GPU":
      os.environ["CUDA_VISIBLE_DEVICES"] = str(_device_id)

    base_opt = de.DynamicEmbeddingOptimizer(base_opt, synchronous=True)
    test_opt = hvd.DistributedOptimizer(test_opt)
    init = tf.keras.initializers.Zeros()
    batch_size = 8
    start = 0
    for dtype, run_step, dim in itertools.product([dtypes.float32], [10], [10]):
      print("device=", "/{}:{}".format(_device, _device_id))
      with tf.device("/{}:{}".format(_device, _device_id)):
        for i in range(1, run_step):
          x = math_ops.range(start, start + batch_size * i, dtype=dtypes.int64)
          x = tf.reshape(x, (batch_size, -1))
          start += batch_size * i
          y = tf.zeros((batch_size, 1), dtype=dtypes.float32)

          base_model = get_emb_sequential_model(
              de.keras.layers.HvdAllToAllEmbedding,
              base_opt,
              embedding_size=dim,
              initializer=init,
              bp_v2=False,
              name='all2all_emb')
          test_model = get_emb_sequential_model(tf.keras.layers.Embedding,
                                                test_opt,
                                                input_dim=start +
                                                batch_size * i,
                                                output_dim=dim,
                                                embeddings_initializer=init,
                                                name='tf_emb')

          base_model.fit(x, y, verbose=0)
          test_model.fit(x, y, verbose=0)

        self.assertAllCloseAccordingToType(
            base_model.layers[1].weights[0],
            test_model.layers[1].weights[0],
            msg="Cond:{},{},{}".format(dtype, run_step, dim),
        )

        self.assertAllCloseAccordingToType(
            base_model.layers[2].weights[0],
            test_model.layers[2].weights[0],
            msg="Cond:{},{},{}".format(dtype, run_step, dim),
        )


if __name__ == "__main__":
  test.main()
