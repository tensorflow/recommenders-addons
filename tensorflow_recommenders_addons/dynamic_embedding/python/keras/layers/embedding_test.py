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
"""unit tests of keras layer APIs
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import itertools
import tensorflow as tf
import tempfile

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf.saved_model_pb2 import SavedModel
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam
try:
  from tensorflow.keras.optimizers.legacy import Adam
except:
  from tensorflow.keras.optimizers import Adam

tf.config.set_soft_device_placement(True)


def get_sequential_model(emb_t, *args, **kwargs):
  l0 = tf.keras.layers.InputLayer(input_shape=(None,), dtype=dtypes.int64)
  l1 = emb_t(*args, **kwargs)
  l2 = tf.keras.layers.Dense(8, 'relu')
  l3 = tf.keras.layers.Dense(1, 'sigmoid')
  if emb_t == de.keras.layers.Embedding:
    model = tf.keras.Sequential([l0, l1, l2, l3])
  elif emb_t == de.keras.layers.FieldWiseEmbedding:
    model = tf.keras.Sequential([l0, l1, tf.keras.layers.Flatten(), l2, l3])
  else:
    raise TypeError('Unsupported embedding layer {}'.format(emb_t))
  return model


default_config = config_pb2.ConfigProto(
    allow_soft_placement=True,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))


@test_util.run_all_in_graph_and_eager_modes
class EmbeddingLayerTest(test.TestCase):

  def test_create(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode')

    dims = [1, 4]
    key_dtypes = [dtypes.int32, dtypes.int64]
    if test_util.is_gpu_available():
      key_dtypes = [dtypes.int64]

    value_dtypes = [dtypes.float32, dtypes.float64]
    if test_util.is_gpu_available():
      value_dtypes = [dtypes.float32]
    initializers = [
        tf.keras.initializers.RandomNormal(),
        tf.keras.initializers.RandomUniform()
    ]
    trainable_options = [True, False]
    bp_options = [True, False]
    restrict_policies = [
        None, de.TimestampRestrictPolicy, de.FrequencyRestrictPolicy
    ]

    rnd = 0

    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      for comb in itertools.product(dims, key_dtypes, value_dtypes,
                                    initializers, trainable_options, bp_options,
                                    restrict_policies):
        name = 'test_creation' + str(rnd)
        embedding = de.keras.layers.Embedding(comb[0],
                                              key_dtype=comb[1],
                                              value_dtype=comb[2],
                                              initializer=comb[3](
                                                  (1,), dtype=comb[2]),
                                              trainable=comb[4],
                                              bp_v2=comb[5],
                                              restrict_policy=comb[6],
                                              init_capacity=64,
                                              name=name)
        rnd += 1
        self.assertAllEqual(embedding.name, name)
        self.assertAllEqual(self.evaluate(embedding.params.size()), 0)

  def test_forward(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode')
    de_init = tf.keras.initializers.RandomNormal(seed=0)
    dense_init = tf.keras.initializers.Ones()
    de_layer = de.keras.layers.Embedding(4, initializer=de_init, name='ve820')
    tf_layer = tf.keras.layers.Embedding(1000,
                                         4,
                                         embeddings_initializer=dense_init,
                                         name='mt047')

    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      ids = math_ops.range(200, dtype=dtypes.int64)
      prepared_values = array_ops.ones((200, 4), dtype=tf.float32)
      self.evaluate(de_layer.params.upsert(ids, prepared_values))
      ids = tf.reshape(ids, (25, 8))
      self.evaluate(variables.global_variables_initializer())
      embedding_value = self.evaluate(de_layer(ids))
      expected_value = self.evaluate(tf_layer(ids))
      self.assertAllClose(embedding_value, expected_value, rtol=1e-6, atol=1e-7)

  def test_backward(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode')
    init = tf.keras.initializers.RandomNormal(seed=0)
    model = get_sequential_model(de.keras.layers.Embedding,
                                 4,
                                 initializer=init,
                                 bp_v2=False,
                                 name='go582')
    optmz = adam.AdamOptimizer(1E-4)
    optmz = de.DynamicEmbeddingOptimizer(optmz)
    emb_layer = model.layers[0]
    model.compile(optimizer=optmz, loss='binary_crossentropy')
    start = 0
    batch_size = 10
    for i in range(1, 10):
      x = math_ops.range(start, start + batch_size * i, dtype=dtypes.int64)
      x = tf.reshape(x, (batch_size, -1))
      start += batch_size * i
      y = tf.zeros((batch_size, 1), dtype=dtypes.float32)
      model.fit(x, y, verbose=0)
      self.assertAllEqual(emb_layer.params.size(), start)

  def test_backward_bp_v2(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode')
    init = tf.keras.initializers.RandomNormal(seed=0)
    model = get_sequential_model(de.keras.layers.Embedding,
                                 4,
                                 initializer=init,
                                 bp_v2=False,
                                 name='iu702')
    optmz = Adam(learning_rate=1E-4, amsgrad=True)
    optmz = de.DynamicEmbeddingOptimizer(optmz)
    emb_layer = model.layers[0]
    model.compile(optimizer=optmz, loss='binary_crossentropy')
    start = 0
    batch_size = 10
    for i in range(1, 10):
      x = math_ops.range(start, start + batch_size * i, dtype=dtypes.int64)
      x = tf.reshape(x, (batch_size, -1))
      start += batch_size * i
      y = tf.zeros((batch_size, 1), dtype=dtypes.float32)
      model.fit(x, y, verbose=0)
      self.assertAllEqual(emb_layer.params.size(), start)

  def test_keras_save_load_weights(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode')
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    def model_fn(table_device):
      input_tensor = tf.keras.layers.Input(shape=(1,), dtype=tf.int64)
      embedding_out = de.keras.layers.Embedding(
          embedding_size=1,
          key_dtype=tf.int64,
          value_dtype=tf.float32,
          initializer=tf.keras.initializers.RandomNormal(),
          devices=table_device,
          name='test_keras_save_restore',
      )(input_tensor)
      model = tf.keras.Model(inputs=input_tensor, outputs=embedding_out)
      optimizer = Adam(learning_rate=1E-4, amsgrad=False)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)
      model.compile(optimizer=optimizer)
      return model

    table_device_ = ['/device:CPU:0']
    if test_util.is_gpu_available():
      table_device_ = ['/device:GPU:0']
    model = model_fn(table_device_)
    params_ = model.get_layer('test_keras_save_restore').params
    params_.upsert(
        constant_op.constant([0, 1], dtypes.int64),
        constant_op.constant([[12.0], [24.0]], dtypes.float32),
    )
    options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
    model.save(save_path, options=options)
    tf.keras.backend.clear_session()
    del model
    model = model_fn(table_device_)
    model.load_weights(save_path).expect_partial()
    params_ = model.get_layer('test_keras_save_restore').params
    size = params_.size()
    self.assertEqual(2, size)
    [keys, values] = params_.export()
    self.assertAllEqual([0, 1], keys)
    self.assertAllEqual([[12.0], [24.0]], values)

    # Check table device was assigned correctly
    graph_path = os.path.join(save_path, 'saved_model.pb')
    sm = SavedModel()
    with open(graph_path, 'rb') as f:
      sm.ParseFromString(f.read())
    for mg in sm.meta_graphs:
      for node in mg.graph_def.node:
        if node.name == 'test_keras_save_restore-parameter_mht_1of1':
          self.assertEqual(table_device_[0], node.device)

  def test_keras_save_load_weights_file_system(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode')
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    def model_fn(table_devices):
      input_tensor = tf.keras.layers.Input(shape=(1,), dtype=tf.int64)
      embedding_outs = []
      for t in range(2):
        embedding_out = de.keras.layers.Embedding(
            embedding_size=1,
            key_dtype=tf.int64,
            value_dtype=tf.float32,
            initializer=tf.keras.initializers.RandomNormal(),
            devices=table_devices,
            name=f'test_keras_save_restore_{t}',
            kv_creator=de.CuckooHashTableCreator(
                saver=de.FileSystemSaver()))(input_tensor)
        embedding_outs.append(embedding_out)
      normal_embedding_out = de.keras.layers.Embedding(
          embedding_size=1,
          key_dtype=tf.int64,
          value_dtype=tf.float32,
          initializer=tf.keras.initializers.RandomNormal(),
          devices=table_devices,
          name='test_keras_save_restore_normal')(input_tensor)
      embedding_outs.append(normal_embedding_out)
      concat = tf.concat(embedding_outs, axis=0)
      model = tf.keras.Model(inputs=input_tensor, outputs=concat)
      optimizer = Adam(learning_rate=1E-4, amsgrad=False)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)
      model.compile(optimizer=optimizer)
      return model

    test_size = 10
    test_keys = [i for i in range(0, test_size)]
    test_values = [[i * 1.0] for i in range(0, test_size)]
    table_device = ['/device:CPU:0']
    if test_util.is_gpu_available():
      table_device = ['/device:GPU:0']
    shard_num = 3
    table_devices_ = table_device * shard_num
    model = model_fn(table_devices_)
    for t in range(2):
      params_ = model.get_layer(f'test_keras_save_restore_{t}').params
      params_.upsert(
          constant_op.constant(test_keys, dtypes.int64),
          constant_op.constant(test_values, dtypes.float32),
      )
    options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
    model.save(save_path, options=options)
    tf.keras.backend.clear_session()
    del model
    model = model_fn(table_devices_)
    model.load_weights(save_path).expect_partial()
    for t in range(2):
      params_ = model.get_layer(f'test_keras_save_restore_{t}').params
      size = params_.size()
      self.assertEqual(test_size, size)
      [keys, values] = params_.export()
      self.assertAllEqual(test_keys, np.sort(keys, axis=0))
      self.assertAllEqual(test_values, np.sort(values, axis=0))

    # test expand shards number
    tf.keras.backend.clear_session()
    del model
    shard_num = 5
    table_devices_ = table_device * shard_num
    model = model_fn(table_devices_)
    model.load_weights(save_path).expect_partial()
    for t in range(2):
      params_ = model.get_layer(f'test_keras_save_restore_{t}').params
      size = params_.size()
      self.assertEqual(test_size, size)
      [keys, values] = params_.export()
      self.assertAllEqual(test_keys, np.sort(keys, axis=0))
      self.assertAllEqual(test_values, np.sort(values, axis=0))

    # test contracte shards number
    tf.keras.backend.clear_session()
    del model
    shard_num = 2
    table_devices_ = table_device * shard_num
    model = model_fn(table_devices_)
    model.load_weights(save_path).expect_partial()
    for t in range(2):
      params_ = model.get_layer(f'test_keras_save_restore_{t}').params
      size = params_.size()
      self.assertEqual(test_size, size)
      [keys, values] = params_.export()
      self.assertAllEqual(test_keys, np.sort(keys, axis=0))
      self.assertAllEqual(test_values, np.sort(values, axis=0))

    # test load all into one shard
    tf.keras.backend.clear_session()
    del model
    shard_num = 1
    table_devices_ = table_device * shard_num
    model = model_fn(table_devices_)
    model.load_weights(save_path).expect_partial()
    for t in range(2):
      params_ = model.get_layer(f'test_keras_save_restore_{t}').params
      size = params_.size()
      self.assertEqual(test_size, size)
      [keys, values] = params_.export()
      self.assertAllEqual(test_keys, np.sort(keys, axis=0))
      self.assertAllEqual(test_values, np.sort(values, axis=0))

  def test_mpi_keras_save_load_weights_file_system(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode')
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")
    de_dir = os.path.join(save_path, "variables", "TFRADynamicEmbedding")

    def model_fn(proc_size, proc_rank):
      table_device = ['/device:CPU:0']
      if test_util.is_gpu_available():
        table_device = ['/device:GPU:0']
      input_tensor = tf.keras.layers.Input(shape=(1,), dtype=tf.int64)
      embedding_outs = []
      for t in range(2):
        embedding_out = de.keras.layers.Embedding(
            embedding_size=1,
            key_dtype=tf.int64,
            value_dtype=tf.float32,
            initializer=tf.keras.initializers.RandomNormal(),
            devices=table_device,
            name=f'test_keras_save_restore_{t}',
            kv_creator=de.CuckooHashTableCreator(saver=de.FileSystemSaver(
                proc_size=proc_size, proc_rank=proc_rank)))(input_tensor)
        embedding_outs.append(embedding_out)
      normal_embedding_out = de.keras.layers.Embedding(
          embedding_size=1,
          key_dtype=tf.int64,
          value_dtype=tf.float32,
          initializer=tf.keras.initializers.RandomNormal(),
          devices=table_device,
          name='test_keras_save_restore_normal')(input_tensor)
      embedding_outs.append(normal_embedding_out)
      concat = tf.concat(embedding_outs, axis=0)
      model = tf.keras.Model(inputs=input_tensor, outputs=concat)
      optimizer = Adam(learning_rate=1E-4, amsgrad=False)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)
      model.compile(optimizer=optimizer)
      return model

    test_size = 30
    test_keys = [i for i in range(0, test_size)]
    test_values = [[i * 1.0] for i in range(0, test_size)]
    table_device = ['/device:CPU:0']
    if test_util.is_gpu_available():
      table_device = ['/device:GPU:0']

    # test same shards number
    proc_size = 3
    keys_shard_size = int(test_size / proc_size)
    models = []
    for i in range(proc_size):
      tf.keras.backend.clear_session()
      models.append(model_fn(proc_size, i))
      for t in range(2):
        params_ = models[i].get_layer(f'test_keras_save_restore_{t}').params
        test_keys_i = test_keys[(i * keys_shard_size):((i + 1) *
                                                       keys_shard_size)]
        test_values_i = test_values[(i * keys_shard_size):((i + 1) *
                                                           keys_shard_size)]
        params_.upsert(
            constant_op.constant(test_keys_i, dtypes.int64),
            constant_op.constant(test_values_i, dtypes.float32),
        )
      options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
      if i == 0:
        models[i].save(save_path, options=options)
      else:
        for t in range(2):
          models[i].get_layer(
              f'test_keras_save_restore_{t}').params.save_to_file_system(
                  dirpath=de_dir, proc_size=proc_size, proc_rank=i)
    tf.keras.backend.clear_session()
    for i in range(proc_size):
      del models[0]
    for i in range(proc_size):
      tf.keras.backend.clear_session()
      models.append(model_fn(proc_size, i))
      models[i].load_weights(save_path).expect_partial()
    for t in range(2):
      total_size = 0
      total_keys = []
      total_values = []
      for i in range(proc_size):
        params_ = models[i].get_layer(f'test_keras_save_restore_{t}').params
        size_i = params_.size()
        total_size = total_size + size_i
        keys, values = params_.export()
        total_keys.extend(keys)
        total_values.extend(values)
      self.assertEqual(test_size, total_size)
      self.assertAllEqual(test_keys, np.sort(total_keys, axis=0))
      self.assertAllEqual(test_values, np.sort(total_values, axis=0))

    # test expand shards number
    proc_size = 3
    keys_shard_size = int(test_size / proc_size)
    models = []
    for i in range(proc_size):
      tf.keras.backend.clear_session()
      models.append(model_fn(proc_size, i))
      for t in range(2):
        params_ = models[i].get_layer(f'test_keras_save_restore_{t}').params
        test_keys_i = test_keys[(i * keys_shard_size):((i + 1) *
                                                       keys_shard_size)]
        test_values_i = test_values[(i * keys_shard_size):((i + 1) *
                                                           keys_shard_size)]
        params_.upsert(
            constant_op.constant(test_keys_i, dtypes.int64),
            constant_op.constant(test_values_i, dtypes.float32),
        )
      options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
      if i == 0:
        models[i].save(save_path, options=options)
      else:
        for t in range(2):
          models[i].get_layer(
              f'test_keras_save_restore_{t}').params.save_to_file_system(
                  dirpath=de_dir, proc_size=proc_size, proc_rank=i)
    tf.keras.backend.clear_session()
    for i in range(proc_size):
      del models[0]
    proc_size = 5
    for i in range(proc_size):
      tf.keras.backend.clear_session()
      models.append(model_fn(proc_size, i))
      models[i].load_weights(save_path).expect_partial()
    for t in range(2):
      total_size = 0
      total_keys = []
      total_values = []
      for i in range(proc_size):
        params_ = models[i].get_layer(f'test_keras_save_restore_{t}').params
        size_i = params_.size()
        total_size = total_size + size_i
        keys, values = params_.export()
        total_keys.extend(keys)
        total_values.extend(values)
      self.assertEqual(test_size, total_size)
      self.assertAllEqual(test_keys, np.sort(total_keys, axis=0))
      self.assertAllEqual(test_values, np.sort(total_values, axis=0))

    # test contracte shards number
    proc_size = 3
    keys_shard_size = int(test_size / proc_size)
    models = []
    for i in range(proc_size):
      tf.keras.backend.clear_session()
      models.append(model_fn(proc_size, i))
      for t in range(2):
        params_ = models[i].get_layer(f'test_keras_save_restore_{t}').params
        test_keys_i = test_keys[(i * keys_shard_size):((i + 1) *
                                                       keys_shard_size)]
        test_values_i = test_values[(i * keys_shard_size):((i + 1) *
                                                           keys_shard_size)]
        params_.upsert(
            constant_op.constant(test_keys_i, dtypes.int64),
            constant_op.constant(test_values_i, dtypes.float32),
        )
      options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
      if i == 0:
        models[i].save(save_path, options=options)
      else:
        for t in range(2):
          models[i].get_layer(
              f'test_keras_save_restore_{t}').params.save_to_file_system(
                  dirpath=de_dir, proc_size=proc_size, proc_rank=i)
    tf.keras.backend.clear_session()
    for i in range(proc_size):
      del models[0]
    for i in range(proc_size):
      tf.keras.backend.clear_session()
      models.append(model_fn(proc_size, i))
      models[i].load_weights(save_path).expect_partial()
    for t in range(2):
      total_size = 0
      total_keys = []
      total_values = []
      for i in range(proc_size):
        params_ = models[i].get_layer(f'test_keras_save_restore_{t}').params
        size_i = params_.size()
        total_size = total_size + size_i
        keys, values = params_.export()
        total_keys.extend(keys)
        total_values.extend(values)
      self.assertEqual(test_size, total_size)
      self.assertAllEqual(test_keys, np.sort(total_keys, axis=0))
      self.assertAllEqual(test_values, np.sort(total_values, axis=0))

    # test expand shards number
    proc_size = 3
    keys_shard_size = int(test_size / proc_size)
    models = []
    for i in range(proc_size):
      tf.keras.backend.clear_session()
      models.append(model_fn(proc_size, i))
      for t in range(2):
        params_ = models[i].get_layer(f'test_keras_save_restore_{t}').params
        test_keys_i = test_keys[(i * keys_shard_size):((i + 1) *
                                                       keys_shard_size)]
        test_values_i = test_values[(i * keys_shard_size):((i + 1) *
                                                           keys_shard_size)]
        params_.upsert(
            constant_op.constant(test_keys_i, dtypes.int64),
            constant_op.constant(test_values_i, dtypes.float32),
        )
      options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
      if i == 0:
        models[i].save(save_path, options=options)
      else:
        for t in range(2):
          models[i].get_layer(
              f'test_keras_save_restore_{t}').params.save_to_file_system(
                  dirpath=de_dir, proc_size=proc_size, proc_rank=i)
    tf.keras.backend.clear_session()
    for i in range(proc_size):
      del models[0]
    proc_size = 2
    for i in range(proc_size):
      tf.keras.backend.clear_session()
      models.append(model_fn(proc_size, i))
      models[i].load_weights(save_path).expect_partial()
    for t in range(2):
      total_size = 0
      total_keys = []
      total_values = []
      for i in range(proc_size):
        params_ = models[i].get_layer(f'test_keras_save_restore_{t}').params
        size_i = params_.size()
        total_size = total_size + size_i
        keys, values = params_.export()
        total_keys.extend(keys)
        total_values.extend(values)
      self.assertEqual(test_size, total_size)
      self.assertAllEqual(test_keys, np.sort(total_keys, axis=0))
      self.assertAllEqual(test_values, np.sort(total_values, axis=0))

    # test load all into one shard
    proc_size = 3
    keys_shard_size = int(test_size / proc_size)
    models = []
    for i in range(proc_size):
      tf.keras.backend.clear_session()
      models.append(model_fn(proc_size, i))
      for t in range(2):
        params_ = models[i].get_layer(f'test_keras_save_restore_{t}').params
        test_keys_i = test_keys[(i * keys_shard_size):((i + 1) *
                                                       keys_shard_size)]
        test_values_i = test_values[(i * keys_shard_size):((i + 1) *
                                                           keys_shard_size)]
        params_.upsert(
            constant_op.constant(test_keys_i, dtypes.int64),
            constant_op.constant(test_values_i, dtypes.float32),
        )
      options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
      if i == 0:
        models[i].save(save_path, options=options)
      else:
        for t in range(2):
          models[i].get_layer(
              f'test_keras_save_restore_{t}').params.save_to_file_system(
                  dirpath=de_dir, proc_size=proc_size, proc_rank=i)
    tf.keras.backend.clear_session()
    for i in range(proc_size):
      del models[0]
    for i in range(proc_size):
      tf.keras.backend.clear_session()
      models.append(model_fn(proc_size, i))
      models[i].load_weights(save_path).expect_partial()
    for t in range(2):
      total_size = 0
      total_keys = []
      total_values = []
      for i in range(proc_size):
        params_ = models[i].get_layer(f'test_keras_save_restore_{t}').params
        size_i = params_.size()
        total_size = total_size + size_i
        keys, values = params_.export()
        total_keys.extend(keys)
        total_values.extend(values)
      self.assertEqual(test_size, total_size)
      self.assertAllEqual(test_keys, np.sort(total_keys, axis=0))
      self.assertAllEqual(test_values, np.sort(total_values, axis=0))

    # test expand shards number
    proc_size = 3
    keys_shard_size = int(test_size / proc_size)
    models = []
    for i in range(proc_size):
      tf.keras.backend.clear_session()
      models.append(model_fn(proc_size, i))
      for t in range(2):
        params_ = models[i].get_layer(f'test_keras_save_restore_{t}').params
        test_keys_i = test_keys[(i * keys_shard_size):((i + 1) *
                                                       keys_shard_size)]
        test_values_i = test_values[(i * keys_shard_size):((i + 1) *
                                                           keys_shard_size)]
        params_.upsert(
            constant_op.constant(test_keys_i, dtypes.int64),
            constant_op.constant(test_values_i, dtypes.float32),
        )
      options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
      if i == 0:
        models[i].save(save_path, options=options)
      else:
        for t in range(2):
          models[i].get_layer(
              f'test_keras_save_restore_{t}').params.save_to_file_system(
                  dirpath=de_dir, proc_size=proc_size, proc_rank=i)
    tf.keras.backend.clear_session()
    for i in range(proc_size):
      del models[0]
    proc_size = 1
    total_size = 0
    total_keys = []
    total_values = []
    for i in range(proc_size):
      tf.keras.backend.clear_session()
      models.append(model_fn(proc_size, i))
      models[i].load_weights(save_path).expect_partial()
    for t in range(2):
      total_size = 0
      total_keys = []
      total_values = []
      for i in range(proc_size):
        params_ = models[i].get_layer(f'test_keras_save_restore_{t}').params
        size_i = params_.size()
        total_size = total_size + size_i
        keys, values = params_.export()
        total_keys.extend(keys)
        total_values.extend(values)
      self.assertEqual(test_size, total_size)
      self.assertAllEqual(test_keys, np.sort(total_keys, axis=0))
      self.assertAllEqual(test_values, np.sort(total_values, axis=0))


@test_util.run_all_in_graph_and_eager_modes
class SquashedEmbeddingLayerTest(test.TestCase):

  def test_forward(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode')
    init = tf.keras.initializers.Zeros()
    de_layer = de.keras.layers.SquashedEmbedding(2,
                                                 initializer=init,
                                                 key_dtype=dtypes.int32,
                                                 value_dtype=dtypes.float32,
                                                 name='tr423')
    dense_init = tf.keras.initializers.Ones()
    tf_layer = tf.keras.layers.Embedding(100,
                                         2,
                                         embeddings_initializer=dense_init,
                                         name='mt047')

    preset_ids = constant_op.constant([3, 0, 1], dtype=dtypes.int32)
    preset_values = constant_op.constant([[1, 1], [1, 1], [1, 1]],
                                         dtype=dtypes.float32)
    de_layer.params.upsert(preset_ids, preset_values)
    de_ids = constant_op.constant([3, 0, 1, 2], dtype=tf.int32)
    output = de_layer(de_ids)
    tf_ids = constant_op.constant([3, 0, 1], dtype=tf.int32)
    expected = tf_layer(tf_ids)
    expected = tf.reduce_sum(expected, axis=0)
    self.assertAllEqual(output, expected)


@test_util.run_all_in_graph_and_eager_modes
class FieldWiseEmbeddingLayerTest(test.TestCase):

  def test_create(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode')

    dims = [1, 4]
    key_dtypes = [dtypes.int32, dtypes.int64]
    if test_util.is_gpu_available():
      key_dtypes = [dtypes.int64]

    value_dtypes = [dtypes.float32, dtypes.float64]
    if test_util.is_gpu_available():
      value_dtypes = [dtypes.float32]
    initializers = [
        tf.keras.initializers.RandomNormal(),
        tf.keras.initializers.RandomUniform()
    ]
    trainable_options = [True, False]
    bp_options = [True, False]
    restrict_policies = [
        None, de.TimestampRestrictPolicy, de.FrequencyRestrictPolicy
    ]

    rnd = 0
    nslots = 100

    @tf.function
    def slot_map_fn(x):
      x = tf.as_string(x)
      x = tf.strings.to_hash_bucket_fast(x, nslots)
      return x

    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      for comb in itertools.product(dims, key_dtypes, value_dtypes,
                                    initializers, trainable_options, bp_options,
                                    restrict_policies):
        name = 'test_creation' + str(rnd)
        embedding = de.keras.layers.FieldWiseEmbedding(comb[0],
                                                       nslots,
                                                       slot_map_fn,
                                                       key_dtype=comb[1],
                                                       value_dtype=comb[2],
                                                       initializer=comb[3](
                                                           (1,), dtype=comb[2]),
                                                       trainable=comb[4],
                                                       bp_v2=comb[5],
                                                       restrict_policy=comb[6],
                                                       init_capacity=64,
                                                       name=name)
        rnd += 1
        self.assertAllEqual(embedding.name, name)
        self.assertAllEqual(self.evaluate(embedding.params.size()), 0)

  def test_forward(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode')

    init = tf.keras.initializers.RandomNormal(seed=0)
    ids = math_ops.range(200, dtype=dtypes.int64)
    ids = tf.reshape(ids, (25, 8))

    def slot_map_fn(x):
      return tf.math.floormod(x, 2)

    de_layer = de.keras.layers.FieldWiseEmbedding(4,
                                                  2,
                                                  slot_map_fn,
                                                  initializer=init,
                                                  name='fr010')
    tf_layer = tf.keras.layers.Embedding(1000,
                                         4,
                                         embeddings_initializer=init,
                                         name='xz774')
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      self.evaluate(variables.global_variables_initializer())
      embedding_value = self.evaluate(de_layer(ids))
      self.assertAllEqual(embedding_value.shape, (25, 2, 4))

  def test_backward(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode')

    @tf.function
    def slot_map_fn(x):
      return tf.math.floormod(x, 2)

    init = tf.keras.initializers.RandomNormal(seed=0)
    model = get_sequential_model(de.keras.layers.FieldWiseEmbedding,
                                 4,
                                 2,
                                 slot_map_fn,
                                 bp_v2=True,
                                 initializer=init,
                                 name='oe423')
    optmz = Adam(learning_rate=1E-4, amsgrad=True)
    optmz = de.DynamicEmbeddingOptimizer(optmz)
    emb_layer = model.layers[0]
    model.compile(optimizer=optmz, loss='binary_crossentropy')
    start = 0
    batch_size = 10
    for i in range(1, 10):
      x = math_ops.range(start, start + batch_size * i, dtype=dtypes.int64)
      x = tf.reshape(x, (batch_size, -1))
      start += batch_size * i
      y = tf.zeros((batch_size, 1), dtype=dtypes.float32)
      model.fit(x, y, verbose=0)
      self.assertAllEqual(emb_layer.params.size(), start)

  def test_sequential_model_save_and_load_weights(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    @tf.function
    def slot_map_fn(x):
      return tf.math.floormod(x, 2)

    init = tf.keras.initializers.RandomNormal(seed=0)
    model = get_sequential_model(de.keras.layers.FieldWiseEmbedding,
                                 4,
                                 2,
                                 slot_map_fn,
                                 bp_v2=False,
                                 initializer=init,
                                 restrict_policy=de.FrequencyRestrictPolicy,
                                 name='pc053')
    optmz = Adam(learning_rate=1E-2, amsgrad=True)
    optmz = de.DynamicEmbeddingOptimizer(optmz)
    emb_layer = model.layers[0]
    model.compile(optimizer=optmz, loss='binary_crossentropy')
    start = 0
    batch_size = 10
    for i in range(1, 10):
      x = math_ops.range(start, start + batch_size * i, dtype=dtypes.int64)
      x = tf.reshape(x, (batch_size, -1))
      start += batch_size * i
      y = tf.zeros((batch_size, 1), dtype=dtypes.float32)
      model.fit(x, y, verbose=0)

    ids = tf.range(0, 10, dtype=tf.int64)
    ids = tf.reshape(ids, (1, -1))
    expected = model(ids)

    save_dir = tempfile.mkdtemp(prefix='/tmp/')
    options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
    model.save(save_dir, signatures=None, options=options)
    copied_init = tf.keras.initializers.RandomNormal(seed=0)
    new_model = get_sequential_model(de.keras.layers.FieldWiseEmbedding,
                                     4,
                                     2,
                                     slot_map_fn,
                                     bp_v2=False,
                                     initializer=copied_init,
                                     restrict_policy=de.FrequencyRestrictPolicy,
                                     name='pc053')
    new_emb_layer = new_model.layers[0]
    new_model.load_weights(save_dir).expect_partial()
    evaluate = new_model(ids)
    self.assertAllEqual(evaluate, expected)

    keys, values = emb_layer.params.export()
    seq = tf.argsort(keys)
    keys = tf.sort(keys)
    values = tf.gather(values, seq)

    rp_keys, rp_values = emb_layer.params.restrict_policy.freq_var.export()
    rp_seq = tf.argsort(rp_keys)
    rp_keys = tf.sort(rp_keys)
    rp_values = tf.gather(rp_values, rp_seq)

    new_keys, new_values = new_emb_layer.params.export()
    seq = tf.argsort(new_keys)
    new_keys = tf.sort(new_keys)
    new_values = tf.gather(new_values, seq)

    new_rp_keys, new_rp_values = \
      new_emb_layer.params.restrict_policy.freq_var.export()
    new_rp_seq = tf.argsort(new_rp_keys)
    new_rp_keys = tf.sort(new_rp_keys)
    new_rp_values = tf.gather(new_rp_values, new_rp_seq)

    self.assertAllEqual(keys, new_keys)
    self.assertAllEqual(rp_keys, new_rp_keys)
    self.assertAllEqual(values, new_values)
    self.assertAllEqual(rp_values, new_rp_values)

  def test_model_save_and_load(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    @tf.function
    def slot_map_fn(x):
      return tf.math.floormod(x, 2)

    init = tf.keras.initializers.RandomNormal(seed=0)

    class MyModel(tf.keras.Model):

      def __init__(self):
        super(MyModel, self).__init__()
        self.l0 = tf.keras.layers.InputLayer(input_shape=(None,),
                                             dtype=tf.int64)
        self.l1 = de.keras.layers.FieldWiseEmbedding(4,
                                                     2,
                                                     slot_map_fn,
                                                     bp_v2=False,
                                                     initializer=init,
                                                     name='sl337')
        self.l2 = tf.keras.layers.Flatten()
        self.l3 = tf.keras.layers.Dense(32, 'relu')
        self.l4 = tf.keras.layers.Dense(1, 'sigmoid')

      def call(self, x):
        return self.l4(self.l3(self.l2(self.l1(self.l0(x)))))

    model = MyModel()

    optmz = Adam(1E-3)
    optmz = de.DynamicEmbeddingOptimizer(optmz)
    model.compile(optimizer=optmz, loss='binary_crossentropy')

    start = 0
    batch_size = 10
    for i in range(1, 10):
      x = math_ops.range(start, start + batch_size * i, dtype=dtypes.int64)
      x = tf.reshape(x, (batch_size, -1))
      start += batch_size * i
      y = tf.zeros((batch_size, 1), dtype=dtypes.float32)
      model.fit(x, y, verbose=0)

    ids = tf.range(0, 10, dtype=tf.int64)
    ids = tf.reshape(ids, (2, -1))

    expected = model(ids)
    save_dir = tempfile.mkdtemp(prefix='/tmp/')
    options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])

    @tf.function(input_signature=[tf.TensorSpec((None, None), dtype=tf.int64)])
    def foo(x):
      return model(x)

    model.save(save_dir, signatures=foo, options=options)
    new_model = tf.saved_model.load(save_dir)
    sig = new_model.signatures['serving_default']
    evaluated = sig(ids)['output_0']
    self.assertAllClose(expected, evaluated, 1E-7, 1E-7)


if __name__ == '__main__':
  test.main()
