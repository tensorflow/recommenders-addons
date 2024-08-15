# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Unit tests of warm-start util"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import itertools
import math
import numpy as np
import os
import shutil

from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.utils.check_platform import is_macos, is_arm64

try:
  from tensorflow.python.keras.initializers import initializers_v2 as kinit2
except ImportError:
  kinit2 = None
  pass  # for compatible with TF < 2.3.x

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import saver
from tensorflow_recommenders_addons import dynamic_embedding as de

import tensorflow as tf


@test_util.deprecated_graph_mode_only
class WarmStartUtilTest(test.TestCase):

  def _test_warm_start(self, num_shards, use_regex):
    devices = ["/cpu:0" for _ in range(num_shards)]
    ckpt_prefix = os.path.join(self.get_temp_dir(), "ckpt")
    id_list = [x for x in range(100)]
    val_list = [[x] for x in range(100)]

    emb_name = "t100_{}_{}".format(num_shards, use_regex)
    with self.session(graph=ops.Graph()) as sess:
      embeddings = de.get_variable(emb_name,
                                   dtypes.int64,
                                   dtypes.float32,
                                   devices=devices,
                                   initializer=0.0)
      ids = constant_op.constant(id_list, dtype=dtypes.int64)
      vals = constant_op.constant(val_list, dtype=dtypes.float32)
      self.evaluate(embeddings.upsert(ids, vals))
      save = saver.Saver(var_list=[embeddings])
      save.save(sess, ckpt_prefix)

    with self.session(graph=ops.Graph()) as sess:
      embeddings = de.get_variable(emb_name,
                                   dtypes.int64,
                                   dtypes.float32,
                                   devices=devices,
                                   initializer=0.0)
      ids = constant_op.constant(id_list, dtype=dtypes.int64)
      emb = de.embedding_lookup(embeddings, ids, name="lookup")
      sess.graph.add_to_collection(de.GraphKeys.DYNAMIC_EMBEDDING_VARIABLES,
                                   embeddings)
      vars_to_warm_start = [embeddings]
      if use_regex:
        vars_to_warm_start = [".*t100.*"]

      restore_op = de.warm_start(ckpt_to_initialize_from=ckpt_prefix,
                                 vars_to_warm_start=vars_to_warm_start)
      self.evaluate(restore_op)
      self.assertAllEqual(emb, val_list)

  def _test_warm_start_rename(self, num_shards, use_regex):
    devices = ["/cpu:0" for _ in range(num_shards)]
    ckpt_prefix = os.path.join(self.get_temp_dir(), "ckpt")
    id_list = [x for x in range(100)]
    val_list = [[x] for x in range(100)]

    emb_name = "t200_{}_{}".format(num_shards, use_regex)
    with self.session(graph=ops.Graph()) as sess:
      embeddings = de.get_variable("save_{}".format(emb_name),
                                   dtypes.int64,
                                   dtypes.float32,
                                   devices=devices,
                                   initializer=0.0)
      ids = constant_op.constant(id_list, dtype=dtypes.int64)
      vals = constant_op.constant(val_list, dtype=dtypes.float32)
      self.evaluate(embeddings.upsert(ids, vals))
      save = saver.Saver(var_list=[embeddings])
      save.save(sess, ckpt_prefix)

    with self.session(graph=ops.Graph()) as sess:
      embeddings = de.get_variable("restore_{}".format(emb_name),
                                   dtypes.int64,
                                   dtypes.float32,
                                   devices=devices,
                                   initializer=0.0)
      ids = constant_op.constant(id_list, dtype=dtypes.int64)
      emb = de.embedding_lookup(embeddings, ids, name="lookup")
      sess.graph.add_to_collection(de.GraphKeys.DYNAMIC_EMBEDDING_VARIABLES,
                                   embeddings)
      vars_to_warm_start = [embeddings]
      if use_regex:
        vars_to_warm_start = [".*t200.*"]

      restore_op = de.warm_start(ckpt_to_initialize_from=ckpt_prefix,
                                 vars_to_warm_start=vars_to_warm_start,
                                 var_name_to_prev_var_name={
                                     "restore_{}".format(emb_name):
                                         "save_{}".format(emb_name)
                                 })
      self.evaluate(restore_op)
      self.assertAllEqual(emb, val_list)

  def _test_warm_start_estimator(self, num_shards, use_regex):
    if (is_macos() and is_arm64()):
      self.skipTest(
          "skip save restore file system test because TFIO doesn't support apple silicon."
      )
    devices = ["/cpu:0" for _ in range(num_shards)]
    ckpt_prefix = os.path.join(self.get_temp_dir(), "ckpt")
    id_list = [x for x in range(100)]
    val_list = [[x] for x in range(100)]

    emb_name = "t300_{}_{}".format(num_shards, use_regex)
    with self.session(graph=ops.Graph()) as sess:
      embeddings = de.get_variable(emb_name,
                                   dtypes.int64,
                                   dtypes.float32,
                                   devices=devices,
                                   initializer=0.0)
      ids = constant_op.constant(id_list, dtype=dtypes.int64)
      vals = constant_op.constant(val_list, dtype=dtypes.float32)
      self.evaluate(embeddings.upsert(ids, vals))
      save = saver.Saver(var_list=[embeddings])
      save.save(sess, ckpt_prefix)

    def _input_fn():
      dataset = tf.data.Dataset.from_tensor_slices({
          'ids':
              constant_op.constant([[x] for x in id_list], dtype=dtypes.int64)
      })
      return dataset

    def _model_fn(features, labels, mode, params):
      ids = features['ids']
      embeddings = de.get_variable(emb_name,
                                   dtypes.int64,
                                   dtypes.float32,
                                   devices=devices,
                                   initializer=0.0)
      emb = de.embedding_lookup(embeddings, ids, name="lookup")
      emb.graph.add_to_collection(de.GraphKeys.DYNAMIC_EMBEDDING_VARIABLES,
                                  embeddings)
      vars_to_warm_start = [embeddings]
      if use_regex:
        vars_to_warm_start = [".*t300.*"]

      warm_start_hook = de.WarmStartHook(ckpt_to_initialize_from=ckpt_prefix,
                                         vars_to_warm_start=vars_to_warm_start)
      return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                        predictions=emb,
                                        prediction_hooks=[warm_start_hook])

    predictor = tf.estimator.Estimator(model_fn=_model_fn)
    predictions = predictor.predict(_input_fn)
    pred_vals = []
    for pred in predictions:
      pred_vals.append(pred)
    self.assertAllEqual(pred_vals, val_list)

  def test_warm_start(self):
    for num_shards in [1, 3]:
      self._test_warm_start(num_shards, True)
      self._test_warm_start(num_shards, False)

  def test_warm_start_rename(self):
    for num_shards in [1, 3]:
      self._test_warm_start_rename(num_shards, True)
      self._test_warm_start_rename(num_shards, False)

  try:  # tf version <= 2.15

    def test_warm_start_estimator(self):
      for num_shards in [1, 3]:
        self._test_warm_start_estimator(num_shards, True)
        self._test_warm_start_estimator(num_shards, False)
  except:
    print(f"estimator is not supported in this version of tensorflow")


if __name__ == "__main__":
  test.main()
