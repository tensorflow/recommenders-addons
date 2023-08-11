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
import os

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.platform import test

import tensorflow as tf
try:
  import tensorflow_io
except:
  print()

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
        if not use_gpu:
          self.assertTrue("_size={}".format(expect_size) in printed.contents())
        else:
          if init_size == 0:
            self.assertTrue(
                "init capacity: {}".format(1024 * 1024) in printed.contents())
          else:
            self.assertTrue(
                "init capacity: {}".format(init_size) in printed.contents())

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
                                  init_size=168)
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
  def test_cuckoo_hashtable_save_file_system(self):
    self.skipTest('Only test for file_system export, need file_system path.')
    test_devices = ['/CPU:0']
    if test_util.is_gpu_available():
      test_devices = ['/GPU:0']
    dim = 8
    for idx, device in enumerate(test_devices):
      var1 = de.get_variable('fsv1_' + str(idx),
                             key_dtype=dtypes.int64,
                             value_dtype=dtypes.float32,
                             initializer=init_ops.random_normal_initializer(
                                 0.0, 0.01),
                             devices=[device],
                             dim=dim)
      var2 = de.get_variable('fsv2_' + str(idx),
                             key_dtype=dtypes.int64,
                             value_dtype=dtypes.float32,
                             initializer=init_ops.random_normal_initializer(
                                 0.0, 0.01),
                             devices=[device],
                             dim=dim)
      init_keys = constant_op.constant(list(range(10000)), dtypes.int64)
      init_values = var1.lookup(init_keys)

      os.environ["AWS_ACCESS_KEY_ID"] = "Q3AM3UQ867SPQQA43P2F"
      os.environ[
          "AWS_SECRET_ACCESS_KEY"] = "zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG"
      os.environ["S3_ENDPOINT"] = "https://play.min.io"

      with self.session():
        self.evaluate(var1.upsert(init_keys, init_values))

        np_keys = self.evaluate(init_keys)
        np_values = self.evaluate(init_values)

        dirpath = "s3://test/tfra_embedding"
        self.evaluate(var1.tables[0].save_to_file_system(dirpath,
                                                         file_name='fsv_' +
                                                         str(idx),
                                                         buffer_size=4096))
        self.evaluate(var2.tables[0].load_from_file_system(dirpath,
                                                           file_name='fsv_' +
                                                           str(idx),
                                                           buffer_size=4096))
        load_keys, load_values = self.evaluate(var2.export())
        sort_idx = load_keys.argsort()
        load_keys = load_keys[sort_idx[::1]]
        load_values = load_values[sort_idx[::1]]

        self.assertAllEqual(np_keys, load_keys)
        self.assertAllEqual(np_values, load_values)

  @test_util.run_in_graph_and_eager_modes()
  def test_cuckoo_hashtable_save_local_file_system(self):
    test_devices = ['/CPU:0']
    if test_util.is_gpu_available():
      test_devices = ['/GPU:0']
    dim = 8
    for idx, device in enumerate(test_devices):
      var1 = de.get_variable('lfsv1_' + str(idx),
                             key_dtype=dtypes.int64,
                             value_dtype=dtypes.float32,
                             initializer=init_ops.random_normal_initializer(
                                 0.0, 0.01),
                             devices=[device],
                             dim=dim)
      var2 = de.get_variable('lfsv2_' + str(idx),
                             key_dtype=dtypes.int64,
                             value_dtype=dtypes.float32,
                             initializer=init_ops.random_normal_initializer(
                                 0.0, 0.01),
                             devices=[device],
                             dim=dim)
      init_keys = constant_op.constant(list(range(10000)), dtypes.int64)
      init_values = var1.lookup(init_keys)

      with self.session():
        self.evaluate(var1.upsert(init_keys, init_values))

        np_keys = self.evaluate(init_keys)
        np_values = self.evaluate(init_values)

        dirpath = "file:///tmp/test_local_file_system/tfra_embedding"
        self.evaluate(var1.tables[0].save_to_file_system(dirpath,
                                                         file_name='lfsv_' +
                                                         str(idx),
                                                         buffer_size=4096))
        self.evaluate(var2.tables[0].load_from_file_system(dirpath,
                                                           file_name='lfsv_' +
                                                           str(idx),
                                                           buffer_size=4096))
        load_keys, load_values = self.evaluate(var2.export())
        sort_idx = load_keys.argsort()
        load_keys = load_keys[sort_idx[::1]]
        load_values = load_values[sort_idx[::1]]

        self.assertAllEqual(np_keys, load_keys)
        self.assertAllEqual(np_values, load_values)

  @test_util.run_in_graph_and_eager_modes()
  def test_cuckoo_hashtable_save_and_load_all_with_local_file_system(self):
    test_devices = [['/CPU:0', '/CPU:1']]
    if test_util.is_gpu_available():
      tf.debugging.set_log_device_placement(True)
      gpus = tf.config.list_physical_devices('GPU')
      if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
          tf.config.set_logical_device_configuration(gpus[0], [
              tf.config.experimental.VirtualDeviceConfiguration(
                  memory_limit=1024)
          ])
          logical_gpus = tf.config.list_logical_devices('GPU')
          print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
          # Virtual devices must be set before GPUs have been initialized
          print(e)
      test_devices = [['/GPU:0', '/GPU:1']]
    dim = 8
    for idx, devices in enumerate(test_devices):
      var1 = de.get_variable(
          'lfslav1_' + str(idx),
          key_dtype=tf.int64,
          value_dtype=tf.float32,
          initializer=init_ops.random_normal_initializer(0.0, 0.01),
          devices=devices,
          dim=dim,
      )
      var2 = de.get_variable(
          'lfslav2_' + str(idx),
          key_dtype=tf.int64,
          value_dtype=tf.float32,
          initializer=init_ops.random_normal_initializer(0.0, 0.01),
          devices=devices,
          dim=dim,
      )
      init_keys = tf.range(0, 20000, dtype=tf.int64)
      init_values = var1.lookup(init_keys)

      with self.session():
        self.evaluate(var1.clear())
        self.evaluate(var1.upsert(init_keys[0:10000], init_values[0:10000]))
        self.evaluate(
            var1.upsert(init_keys[10000:20000], init_values[10000:20000]))
        self.evaluate(var2.clear())

        np_keys = self.evaluate(init_keys)
        np_values = self.evaluate(init_values)

        dirpath = "file:///tmp/test_tfra/test"
        self.evaluate(var1.tables[0].save_to_file_system(dirpath,
                                                         buffer_size=1000))
        self.evaluate(var1.tables[1].save_to_file_system(dirpath,
                                                         buffer_size=1000))
        self.evaluate(var2.tables[0].load_from_file_system(
            dirpath,
            file_name='lfslav1_' + str(idx),
            load_entire_dir=True,
            buffer_size=1000))
        load_keys, load_values = self.evaluate(var2.export())
        sort_idx = load_keys.argsort()
        load_keys = load_keys[sort_idx[::1]]
        load_values = load_values[sort_idx[::1]]

        self.assertAllEqual(np_keys, load_keys)
        self.assertAllEqual(np_values, load_values)


if __name__ == "__main__":
  test.main()
