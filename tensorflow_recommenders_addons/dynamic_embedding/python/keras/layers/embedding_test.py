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

import numpy as np
import itertools
import tensorflow as tf
import tempfile

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam

tf.config.set_soft_device_placement(True)


def get_sequential_model(emb_t, *args, **kwargs):
  l0 = tf.keras.layers.InputLayer(input_shape=(None,), dtype=dtypes.int64)
  l1 = emb_t(*args, **kwargs)
  l2 = tf.keras.layers.Dense(8, 'relu')
  l3 = tf.keras.layers.Dense(1, 'sigmoid')
  if emb_t == de.keras.layers.BasicEmbedding:
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
class BasicEmbeddingLayerTest(test.TestCase):

  def test_create(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode')

    dims = [1, 4]
    key_dtypes = [dtypes.int32, dtypes.int64]
    if test_util.is_gpu_available():
      key_dtypes = [dtypes.int64]

    value_dtypes = [dtypes.float32, dtypes.float64]
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
        embedding = de.keras.layers.BasicEmbedding(comb[0],
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
    de_layer = de.keras.layers.BasicEmbedding(4,
                                              initializer=de_init,
                                              name='ve820')
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
    model = get_sequential_model(de.keras.layers.BasicEmbedding,
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
    model = get_sequential_model(de.keras.layers.BasicEmbedding,
                                 4,
                                 initializer=init,
                                 bp_v2=False,
                                 name='iu702')
    optmz = tf.keras.optimizers.Adam(learning_rate=1E-4, amsgrad=True)
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
    optmz = tf.keras.optimizers.Adam(learning_rate=1E-4, amsgrad=True)
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
                                 name='pc053')
    optmz = tf.keras.optimizers.Adam(learning_rate=1E-2, amsgrad=True)
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
                                     name='pc053')
    new_emb_layer = new_model.layers[0]
    new_model.load_weights(save_dir)
    evaluate = new_model(ids)
    self.assertAllEqual(evaluate, expected)

    keys, values = emb_layer.params.export()
    seq = tf.argsort(keys)
    keys = tf.sort(keys)
    values = tf.gather(values, seq)

    new_keys, new_values = new_emb_layer.params.export()
    seq = tf.argsort(new_keys)
    new_keys = tf.sort(new_keys)
    new_values = tf.gather(new_values, seq)

    self.assertAllEqual(keys, new_keys)
    self.assertAllEqual(values, new_values)

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

    optmz = tf.keras.optimizers.Adam(1E-3)
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
