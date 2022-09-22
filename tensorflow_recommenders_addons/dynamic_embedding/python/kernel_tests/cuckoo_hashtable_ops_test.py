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
"""unit tests of cuckoo hashtable ops
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

default_config = config_pb2.ConfigProto(
    allow_soft_placement=False,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))


class CuckooHashtableTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def test_dynamic_embedding_variable_set_init_size(self):
    test_list = [["CPU", False, 12345, 12345], ["CPU", False, 0, 8192]]
    if test_util.is_gpu_available():
      test_list = [["GPU", True, 54321, 54321], ["GPU", True, 0, 8192]]
    id = 0
    for dev_str, use_gpu, init_size, expect_size in test_list:
      with self.session(use_gpu=use_gpu, config=default_config):
        with self.captureWritesToStream(sys.stderr) as printed:
          table = de.get_variable("2021-" + str(id),
                                  dtypes.int64,
                                  dtypes.int32,
                                  initializer=0,
                                  dim=8,
                                  init_size=init_size)
          self.evaluate(table.size())
        id += 1
        self.assertTrue("I" in printed.contents())
        self.assertTrue(dev_str in printed.contents())
        self.assertTrue("_size={}".format(expect_size) in printed.contents())

  @test_util.run_in_graph_and_eager_modes()
  def test_cuckoo_hashtable_import_and_export(self):
    test_list = [["CPU", False]]
    if test_util.is_gpu_available():
      test_list = [["GPU", True]]
    id = 0
    for device, use_gpu in test_list:
      with self.session(use_gpu=use_gpu, config=default_config):
        with self.captureWritesToStream(sys.stderr) as printed:
          table = de.get_variable("2021-" + str(id),
                                  dtypes.int64,
                                  dtypes.int32,
                                  initializer=0,
                                  dim=3,
                                  init_size=128)
          keys = constant_op.constant(list(range(168)), dtypes.int64)
          values = constant_op.constant([[1, 1, 1] for _ in range(168)],
                                        dtypes.int32)
          self.evaluate(table.upsert(keys, values))
          self.assertAllEqual(168, self.evaluate(table.size()))
          exported_keys, exported_values = self.evaluate(table.export())
          exported_keys = sorted(exported_keys)
          self.assertAllEqual(keys, exported_keys)
          self.assertEqual(168, len(exported_values))
          id += 1

  @test_util.run_in_graph_and_eager_modes()
  def test_cuckoo_hashtable_save_hdfs(self):
    self.skipTest('Only test for hdfs export, need hdfs path.')
    initializer = tf.keras.initializers.RandomNormal()
    dim = 8

    test_devices = ['/CPU:0']
    for idx, device in enumerate(test_devices):
      var1 = de.get_variable('vmas142_' + str(idx),
                             key_dtype=tf.int64,
                             value_dtype=tf.float32,
                             initializer=initializer,
                             devices=[device],
                             dim=dim)
      var2 = de.get_variable('lfwa031_' + str(idx),
                             key_dtype=tf.int64,
                             value_dtype=tf.float32,
                             initializer=initializer,
                             devices=[device],
                             dim=dim)
      init_keys = tf.range(0, 10000, dtype=tf.int64)
      init_values = var1.lookup(init_keys)

      with self.session():
        self.evaluate(var1.upsert(init_keys, init_values))

        np_keys = self.evaluate(init_keys)
        np_values = self.evaluate(init_values)

        filepath = "hdfs://path_to_test"
        self.evaluate(var1.tables[0].save_to_hdfs(filepath, buffer_size=4096))
        self.evaluate(var2.tables[0].load_from_hdfs(filepath, buffer_size=4096))
        load_keys, load_values = self.evaluate(var2.export())
        sort_idx = load_keys.argsort()
        load_keys = load_keys[sort_idx[::1]]
        load_values = load_values[sort_idx[::1]]

        self.assertAllEqual(np_keys, load_keys)
        self.assertAllEqual(np_values, load_values)


if __name__ == "__main__":
  test.main()
