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
import numpy as np
import shutil

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
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import monitored_session
from tensorflow.python.training.optimizer import Optimizer as tf1_opt
from tensorflow.python.training import training_util
try:
  from tensorflow.keras.optimizers.legacy import Adam
except:
  from tensorflow.keras.optimizers import Adam
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


is_gpu_available = test_util.is_gpu_available()


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
    keras_base_opt = Adam(1.0)
    keras_test_opt = Adam(1.0)
    self.common_minimize_trainable_v2(keras_base_opt,
                                      keras_test_opt,
                                      name="keras_adam")

  @test_util.run_all_in_graph_and_eager_modes
  def test_all_to_all_embedding_trainable(self):
    # TODO: Resolve the conflict arising from the 'save' function incompatibility with TensorFlow 2.11.
    if (tf.__version__ == "2.11.0" or tf.__version__ == "2.11.1"):
      self.skipTest(
          "The save function doesn't work with TF 2.11, skip the test.")
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    if (is_macos() and is_arm64()):
      self.skipTest(
          "Apple silicon devices don't support synchronous training based on Horovod."
      )
    keras_base_opt = Adam(1.0)
    keras_test_opt = Adam(1.0)
    self.common_all_to_all_embedding_trainable_v2(keras_base_opt,
                                                  keras_test_opt,
                                                  name="keras_adam")
    self.common_lazy_build_model_with_checkpoint_management_v2(
        name="keras_adam_lazy_build")

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
    kv_creator = de.CuckooHashTableCreator(
        saver=de.FileSystemSaver(proc_size=hvd.size(), proc_rank=hvd.rank()))
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
              kv_creator=kv_creator,
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

        a2aemb_size = base_model.layers[0].params.size()
        save_dir = "/tmp/hvd_save_restore" + str(
            hvd.size()) + str(run_step) + str(
                dim)  # All ranks should share same save directory
        save_options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
        if hvd.rank() == 0:
          if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        hvd.join()  # Sync for avoiding files conflict
        # base_model.save(save_dir, options=save_options)
        de.keras.models.de_save_model(base_model,
                                      save_dir,
                                      options=save_options)
        ckpt = de.train.DECheckpoint(
            my_model=base_model)  # Test custom model key "my_model"
        ckpt.save(save_dir + '/ckpt/test')
        del base_model
        del base_opt
        tf.keras.backend.clear_session()
        new_opt = de.DynamicEmbeddingOptimizer(Adam(1.1), synchronous=True)
        new_base_model = get_emb_sequential_model(
            de.keras.layers.HvdAllToAllEmbedding,
            new_opt,
            dense_init='ones',
            embedding_size=dim,
            initializer=init,
            bp_v2=False,
            kv_creator=kv_creator,
            name='all2all_emb')
        ckpt = de.train.DECheckpoint(my_model=new_base_model)
        hvd.join()  # Sync for avoiding files conflict
        ckpt.restore(tf.train.latest_checkpoint(save_dir + '/ckpt/'))
        new_a2aemb_size = new_base_model.layers[0].params.size()
        self.assertEqual(a2aemb_size, new_a2aemb_size)
        hvd.join()  # Sync for avoiding files conflict
        tf.keras.backend.clear_session()
        new_base_model.load_weights(save_dir + '/variables/variables')
        new_a2aemb_size = new_base_model.layers[0].params.size()
        self.assertEqual(a2aemb_size, new_a2aemb_size)
        hvd.join()  # Sync for avoiding files conflict

  def common_lazy_build_model_with_checkpoint_management_v2(self, name):
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

    dim = 8

    class NoCompileModel(tf.keras.models.Model):

      def __init__(self, init, dynamic=False):
        super().__init__(dynamic=dynamic)
        kv_creator = de.CuckooHashTableCreator(saver=de.FileSystemSaver(
            proc_size=hvd.size(), proc_rank=hvd.rank()))
        self.emb = de.keras.layers.HvdAllToAllEmbedding(embedding_size=dim,
                                                        devices=['/GPU:0'],
                                                        initializer=0,
                                                        kv_creator=kv_creator,
                                                        name=name)
        self.l1 = tf.keras.layers.Dense(8, 'relu', kernel_initializer=init)
        self.l2 = tf.keras.layers.Dense(1, 'sigmoid', kernel_initializer=init)

      def build(self, input_shape):
        self.emb.build(input_shape)
        self.l1.build(input_shape + dim)
        self.l2.build(input_shape + 8)

      def call(self, x):
        out = self.emb(x)
        out = self.l1(out)
        return self.l2(out)

    def check_TFRADynamicEmbedding_directory(save_dir,
                                             save_it=None,
                                             should_be_exist=True):
      hvd_size = hvd.size()
      if hvd_size <= 1:
        hvd_size = 1
      base_dir = os.path.join(save_dir, 'variables', 'TFRADynamicEmbedding')
      if save_it is not None:
        base_dir = os.path.join(save_dir, f'TFRADynamicEmbedding-{save_it}')
      for tag in ['keys', 'values']:
        for rank in range(hvd_size):
          self.assertTrue(not (os.path.exists(
              base_dir +
              f'/{name}-parameter_mht_1of1_rank{rank}_size{hvd_size}-{tag}') ^
                               should_be_exist))
          self.assertTrue(not (os.path.exists(
              base_dir +
              f'/{name}-parameter_DynamicEmbedding_keras_adam_lazy_build-shadow_m_mht_1of1_rank{rank}_size{hvd_size}-{tag}'
          ) ^ should_be_exist))
          # f'/{name}-parameter_no_compile_model_DynamicEmbedding_keras_adam_lazy_build-shadow_m_mht_1of1_rank{rank}_size{hvd_size}-{tag}'
          self.assertTrue(not (os.path.exists(
              base_dir +
              f'/{name}-parameter_DynamicEmbedding_keras_adam_lazy_build-shadow_v_mht_1of1_rank{rank}_size{hvd_size}-{tag}'
          ) ^ should_be_exist))
          # f'/{name}-parameter_no_compile_model_DynamicEmbedding_keras_adam_lazy_build-shadow_v_mht_1of1_rank{rank}_size{hvd_size}-{tag}'

    with tf.device("/{}:{}".format(_device, _device_id)):
      x = tf.reshape(tf.range(0, 32, dtype=tf.int64), [32, 1])
      y = tf.random.uniform(shape=[32, 1])
      base_de_emb_standard = {}
      base_de_opt_standard = {}
      new_de_emb_compared = {}
      new_de_opt_compared = {}

      save_dir = self.get_temp_dir()

      model = NoCompileModel('ones')
      base_opt = Adam(1.0)
      base_opt = de.DynamicEmbeddingOptimizer(base_opt, synchronous=True)
      ckpt = de.train.DECheckpoint(model=model, optimizer=base_opt)
      model.compile(optimizer=base_opt, loss='mean_absolute_error')
      manager = checkpoint_management.CheckpointManager(ckpt,
                                                        save_dir,
                                                        max_to_keep=1)
      model.fit(x, y, verbose=0)
      manager.save()
      if hvd.rank() == 0:
        check_TFRADynamicEmbedding_directory(save_dir,
                                             save_it=1,
                                             should_be_exist=True)
      for l in model.layers:
        if name in l.name:
          l.params.upsert(x * 10, tf.random.uniform(shape=[32, 1, dim]))
          emb_size = l.params.size()
          emb_keys, emb_values = l.params.export()
          base_de_emb_standard[l.name] = (emb_size, emb_keys, emb_values)
          break
      for v in base_opt.variables():
        if name in v.name:
          v.params.upsert(x * 10, tf.random.uniform(shape=[32, 1, dim]))
          opt_size = v.params.size()
          opt_keys, opt_values = v.params.export()
          base_de_opt_standard[v._shared_name.split('/')[-1]] = (opt_size,
                                                                 opt_keys,
                                                                 opt_values)
      manager.save()
      if hvd.rank() == 0:
        check_TFRADynamicEmbedding_directory(save_dir,
                                             save_it=2,
                                             should_be_exist=True)
      # CheckpointManager delete checkpoint after the write functuon, but DE KV checkpoint saving and deleting inside the write functuon.
      # So DE KV checkpoint TFRADynamicEmbedding directory will be always one more than TF checkpoint file.
      manager.save()
      if hvd.rank() == 0:
        check_TFRADynamicEmbedding_directory(
            save_dir, save_it=1, should_be_exist=False
        )  # Check delete TFRADynamicEmbedding directory properly.

      del base_opt
      del model
      del ckpt
      tf.keras.backend.clear_session()
      tf.compat.v1.reset_default_graph()

      new_model = NoCompileModel('zeros')
      new_opt = Adam(1.1)
      new_opt = de.DynamicEmbeddingOptimizer(new_opt, synchronous=True)
      new_ckpt = de.train.DECheckpoint(model=new_model, optimizer=new_opt)
      manager = checkpoint_management.CheckpointManager(new_ckpt,
                                                        save_dir,
                                                        max_to_keep=1)
      manager.restore_or_initialize()
      new_model.compile(optimizer=new_opt, loss='mean_absolute_error')
      new_model(x)  # Build vairiables
      try:
        new_opt._create_all_weights([
            new_model.variables[0]
        ])  # Create DE slot variable from DE shadow variable
      except:
        #TODO(MoFHejia) raise ValueError: Cannot convert a partially known TensorShape <unknown> to a Tensor.
        pass
      for l in new_model.layers:
        if name in l.name:
          new_emb_size = l.params.size()
          new_emb_keys, new_emb_values = l.params.export()
          new_de_emb_compared[l.name] = (new_emb_size, new_emb_keys,
                                         new_emb_values)
          break
      for v in new_opt.variables():
        if name in v.name:
          new_opt_size = v.params.size()
          new_opt_keys, new_opt_values = v.params.export()
          new_de_opt_compared[v._shared_name.split('/')[-1]] = (new_opt_size,
                                                                new_opt_keys,
                                                                new_opt_values)

      for de_l_name in base_de_emb_standard.keys():
        self.assertEqual(base_de_emb_standard[de_l_name][0],
                         new_de_emb_compared[de_l_name][0])
        self.assertAllEqual(np.sort(base_de_emb_standard[de_l_name][1], axis=0),
                            np.sort(new_de_emb_compared[de_l_name][1], axis=0))
        self.assertAllClose(np.sort(base_de_emb_standard[de_l_name][2], axis=0),
                            np.sort(new_de_emb_compared[de_l_name][2], axis=0))
      for opt_v_name in base_de_opt_standard.keys():
        self.assertEqual(base_de_opt_standard[opt_v_name][0],
                         new_de_opt_compared[opt_v_name][0])
        self.assertAllEqual(
            np.sort(base_de_opt_standard[opt_v_name][1], axis=0),
            np.sort(new_de_opt_compared[opt_v_name][1], axis=0))
        self.assertAllClose(
            np.sort(base_de_opt_standard[opt_v_name][2], axis=0),
            np.sort(new_de_opt_compared[opt_v_name][2], axis=0))

      extra_save_dir = self.get_temp_dir() + '/extra_save_dir'
      de.keras.models.de_save_model(new_model, extra_save_dir)
      if hvd.rank() == 0:
        check_TFRADynamicEmbedding_directory(extra_save_dir)
      del new_opt
      del new_model
      del new_ckpt
      tf.keras.backend.clear_session()
      tf.compat.v1.reset_default_graph()
      new_saved_model = NoCompileModel('zeros')
      new_saved_opt = Adam(1.2)
      new_saved_opt = de.DynamicEmbeddingOptimizer(new_saved_opt,
                                                   synchronous=True)
      new_saved_model.compile(optimizer=new_saved_opt,
                              loss='mean_absolute_error')
      new_saved_model(x)  # Build vairiables
      try:
        new_opt._create_all_weights([
            new_model.variables[0]
        ])  # Create DE slot variable from DE shadow variable
      except:
        #TODO(MoFHejia) raise ValueError: Cannot convert a partially known TensorShape <unknown> to a Tensor.
        pass
      extra_save_dir = hvd.broadcast_object(
          extra_save_dir, root_rank=0, name='de_utest_hvd_broadcast_filepath'
      )  # All ranks should share same save directory
      new_saved_model.load_weights(extra_save_dir + '/variables/variables')
      for l in new_saved_model.layers:
        if name in l.name:
          new_emb_size = l.params.size()
          new_emb_keys, new_emb_values = l.params.export()
          new_de_emb_compared[l.name] = (new_emb_size, new_emb_keys,
                                         new_emb_values)
          break
      for v in new_saved_opt.variables():
        if name in v.name:
          new_opt_size = v.params.size()
          new_opt_keys, new_opt_values = l.params.export()
          new_de_opt_compared[v._shared_name.split('/')[-1]] = (new_opt_size,
                                                                new_opt_keys,
                                                                new_opt_values)

      for de_l_name in base_de_emb_standard.keys():
        self.assertEqual(base_de_emb_standard[de_l_name][0],
                         new_de_emb_compared[de_l_name][0])
        self.assertAllEqual(np.sort(base_de_emb_standard[de_l_name][1], axis=0),
                            np.sort(new_de_emb_compared[de_l_name][1], axis=0))
        self.assertAllClose(np.sort(base_de_emb_standard[de_l_name][2], axis=0),
                            np.sort(new_de_emb_compared[de_l_name][2], axis=0))
      for opt_v_name in base_de_opt_standard.keys():
        self.assertEqual(base_de_opt_standard[opt_v_name][0],
                         new_de_opt_compared[opt_v_name][0])
        self.assertAllEqual(
            np.sort(base_de_opt_standard[opt_v_name][1], axis=0),
            np.sort(new_de_opt_compared[opt_v_name][1], axis=0))
        self.assertAllClose(
            np.sort(base_de_opt_standard[opt_v_name][2], axis=0),
            np.sort(new_de_opt_compared[opt_v_name][2], axis=0))


if __name__ == "__main__":
  test.main()
