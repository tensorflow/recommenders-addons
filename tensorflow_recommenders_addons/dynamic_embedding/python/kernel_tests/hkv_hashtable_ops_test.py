# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""unit tests of hkv hashtable ops
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import itertools
import numpy as np

from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.utils.check_platform import is_windows, is_macos, is_arm64, is_linux, is_raspi_arm

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
from tensorflow.python.client import session

import tensorflow as tf
try:
  import tensorflow_io
except:
  print()


def _type_converter(tf_type):
  mapper = {
      dtypes.int32: np.int32,
      dtypes.int64: np.int64,
      dtypes.float32: float,
      dtypes.float64: np.float64,
      dtypes.string: str,
      dtypes.half: np.float16,
      dtypes.int8: np.int8,
      dtypes.bool: bool,
  }
  return mapper[tf_type]


default_config = config_pb2.ConfigProto(
    allow_soft_placement=False,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))


def _get_devices():
  return ["/gpu:0" if test_util.is_gpu_available() else "/cpu:0"]


is_gpu_available = test_util.is_gpu_available()


class HkvHashtableTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def test_basic(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    with self.session(use_gpu=True, config=default_config):
      table = de.get_variable(
          'basic',
          key_dtype=dtypes.int64,
          value_dtype=dtypes.int32,
          initializer=0,
          dim=8,
          init_size=1024,
          kv_creator=de.HkvHashTableCreator(config=de.HkvHashTableConfig(
              max_capacity=99999)))
      self.evaluate(table.size())
      self.evaluate(table.clear())
      del table

  def test_variable(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    id = 0
    dim_list = [1, 2, 4, 8, 10, 16, 32, 64, 100, 200]
    kv_list = [[dtypes.int64, dtypes.int8], [dtypes.int64, dtypes.int32],
               [dtypes.int64, dtypes.int64], [dtypes.int64, dtypes.float32],
               [dtypes.int64, dtypes.half]]

    def _convert(v, t):
      return np.array(v).astype(_type_converter(t))

    for (key_dtype, value_dtype), dim in itertools.product(kv_list, dim_list):
      id += 1
      # Skip float16 tests if the platform is macOS arm64 architecture
      if is_macos() and is_arm64():
        if value_dtype == dtypes.half:
          continue
      with self.session(config=default_config, use_gpu=True) as sess:
        keys = constant_op.constant(
            np.array([0, 1, 2, 3]).astype(_type_converter(key_dtype)),
            key_dtype)
        values = constant_op.constant(
            _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype),
            value_dtype)
        table = de.get_variable(
            't1-' + str(id) + '_test_variable',
            key_dtype=key_dtype,
            value_dtype=value_dtype,
            initializer=np.array([-1]).astype(_type_converter(value_dtype)),
            dim=dim,
            init_size=1024,
            kv_creator=de.HkvHashTableCreator(config=de.HkvHashTableConfig(
                max_capacity=99999)))
        table.clear()

        self.assertAllEqual(0, self.evaluate(table.size()))

        self.evaluate(table.upsert(keys, values))
        self.assertAllEqual(4, self.evaluate(table.size()))

        output = table.lookup(keys)
        self.assertAllEqual(values, self.evaluate(output))

        remove_keys = constant_op.constant(_convert([1, 5], key_dtype),
                                           key_dtype)

        self.evaluate(table.remove(remove_keys))

        self.assertAllEqual(3, self.evaluate(table.size()))

        remove_keys = constant_op.constant(_convert([0, 1, 5], key_dtype),
                                           key_dtype)
        output = table.lookup(remove_keys)
        self.assertAllEqual([3, dim], output.get_shape())

        result = self.evaluate(output)
        self.assertAllEqual(
            _convert([[0] * dim, [-1] * dim, [-1] * dim], value_dtype),
            _convert(result, value_dtype))

        exported_keys, exported_values = table.export()

        # exported data is in the order of the internal map, i.e. undefined
        sorted_keys = np.sort(self.evaluate(exported_keys))
        sorted_values = np.sort(self.evaluate(exported_values), axis=0)
        self.assertAllEqual(_convert([0, 2, 3], key_dtype),
                            _convert(sorted_keys, key_dtype))
        self.assertAllEqual(
            _convert([[0] * dim, [2] * dim, [3] * dim], value_dtype),
            _convert(sorted_values, value_dtype))

        self.evaluate(table.clear())
        del table

  def test_dynamic_embedding_variable_set_init_size(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    test_list = [["GPU", True, 54321, 54321], ["GPU", True, 0, 8192]]
    self.assertTrue(len(test_list) > 0)
    id = 0
    for dev_str, use_gpu, init_size, expect_size in test_list:
      with self.session(use_gpu=use_gpu, config=default_config):
        with self.captureWritesToStream(sys.stderr) as printed:
          table = de.get_variable(
              "2021-" + str(id),
              dtypes.int64,
              dtypes.int32,
              initializer=0,
              dim=8,
              init_size=init_size,
              kv_creator=de.HkvHashTableCreator(config=de.HkvHashTableConfig(
                  max_capacity=99999)))
          self.evaluate(table.size())
        id += 1

  def test_hkv_hashtable_import_and_export(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    test_list = [["GPU", True]]
    id = 0
    for device, use_gpu in test_list:
      with self.session(use_gpu=use_gpu, config=default_config):
        with self.captureWritesToStream(sys.stderr) as printed:
          table = de.get_variable(
              "2021-" + str(id),
              dtypes.int64,
              dtypes.int32,
              initializer=0,
              dim=3,
              init_size=128,
              kv_creator=de.HkvHashTableCreator(config=de.HkvHashTableConfig(
                  max_capacity=99999)))
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
  def test_insert(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    dim_list = [1, 2, 4, 8, 10, 16, 32, 64, 100, 200]
    kv_list = [[dtypes.int64, dtypes.float32], [dtypes.int64, dtypes.int32],
               [dtypes.int64, dtypes.int64], [dtypes.int64, dtypes.int8]]

    def _convert(v, t):
      return np.array(v).astype(_type_converter(t))

    id = 0

    for (key_dtype, value_dtype), dim in itertools.product(kv_list, dim_list):
      id += 1
      # Skip float16 tests if the platform is macOS arm64 architecture
      with self.session(config=default_config, use_gpu=True) as sess:
        table = de.get_variable('test_insert' + str(id),
                                key_dtype=key_dtype,
                                value_dtype=value_dtype,
                                initializer=np.array([-1]).astype(
                                    _type_converter(value_dtype)),
                                dim=dim,
                                init_size=102400)

        base_keys = []
        for i in range(18):
          keys = constant_op.constant(
              np.array([x for x in range(85 * i, 85 * (i + 1))
                       ]).astype(_type_converter(key_dtype)), key_dtype)
          values = constant_op.constant(
              _convert([[x] * dim for x in range(85 * i, 85 * (i + 1))],
                       value_dtype), value_dtype)

          self.evaluate(table.upsert(keys, values))
          self.assertAllEqual((i + 1) * 85, self.evaluate(table.size()))
        self.evaluate(table.clear())
        del table

  @test_util.run_in_graph_and_eager_modes()
  def test_hkv_hashtable_save_local_file_system(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    test_devices = ['/GPU:0']
    dim = 8
    for idx, device in enumerate(test_devices):
      var1 = de.get_variable(
          'lfsv1_' + str(idx),
          key_dtype=dtypes.int64,
          value_dtype=dtypes.float32,
          initializer=init_ops.random_normal_initializer(0.0, 0.01),
          devices=[device],
          init_size=3000,
          dim=dim,
          kv_creator=de.HkvHashTableCreator(config=de.HkvHashTableConfig(
              max_capacity=99999)))
      var2 = de.get_variable(
          'lfsv2_' + str(idx),
          key_dtype=dtypes.int64,
          value_dtype=dtypes.float32,
          initializer=init_ops.random_normal_initializer(0.0, 0.01),
          devices=[device],
          init_size=3000,
          dim=dim,
          kv_creator=de.HkvHashTableCreator(config=de.HkvHashTableConfig(
              max_capacity=99999)))
      init_keys = constant_op.constant(list(range(10000)), dtypes.int64)
      init_values = var1.lookup(init_keys)

      with self.session(config=default_config, use_gpu=True):
        self.evaluate(var1.upsert(init_keys, init_values))

        np_keys = self.evaluate(init_keys)
        np_values = self.evaluate(init_values)

        dirpath = "file:///tmp/test_local_file_system/tfra_embedding"
        self.evaluate(var1.tables[0].save_to_file_system(dirpath,
                                                         file_name="test",
                                                         buffer_size=4096))
        self.evaluate(var2.tables[0].load_from_file_system(dirpath,
                                                           file_name="test",
                                                           buffer_size=4096))

        load_keys, load_values = self.evaluate(var2.export())
        sort_idx = load_keys.argsort()
        load_keys = load_keys[sort_idx[::1]]
        load_values = load_values[sort_idx[::1]]

        self.assertAllEqual(np_keys, np.sort(load_keys))
        self.assertAllEqual(np.sort(np_values), np.sort(load_values))

  """
    base kv [0, 1, 2, 3] [0, 1, 2, 3]
    add kv [100] [99]
    remove k [1]
    accum key [0, 1, 100, 3]
    old val [0, 1, 2, 3]
    new val [10 ,11, 100, 13]
    export_exist [t, t, f, t]

      insert base kv
      lookup accum key
      insert add kv
      remove remove k [0, 2, 3, 100] [0, 2, 3, 99]
  """

  def test_variable_find_with_exists_and_accum(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    id = 0
    dim_list = [1, 2, 4, 8, 10, 16, 32, 64, 100, 200]
    kv_list = [[dtypes.int64, dtypes.float32], [dtypes.int64, dtypes.int8],
               [dtypes.int64, dtypes.int32], [dtypes.int64, dtypes.int64],
               [dtypes.int64, dtypes.half]]

    def _convert(v, t):
      return np.array(v).astype(_type_converter(t))

    for (key_dtype, value_dtype), dim in itertools.product(kv_list, dim_list):
      id += 1
      with self.session(config=default_config, use_gpu=True) as sess:
        base_keys = constant_op.constant(
            np.array([0, 1, 2, 3]).astype(_type_converter(key_dtype)),
            key_dtype)
        base_values = constant_op.constant(
            _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype),
            value_dtype)

        simulate_other_process_add_keys = constant_op.constant(
            np.array([100]).astype(_type_converter(key_dtype)), key_dtype)
        simulate_other_process_add_vals = constant_op.constant(
            _convert([
                [99] * dim,
            ], value_dtype), value_dtype)

        simulate_other_process_remove_keys = constant_op.constant(
            np.array([1]).astype(_type_converter(key_dtype)), key_dtype)
        accum_keys = constant_op.constant(
            np.array([0, 1, 100, 3]).astype(_type_converter(key_dtype)),
            key_dtype)
        old_values = constant_op.constant(
            _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype),
            value_dtype)
        new_values = constant_op.constant(
            _convert([[10] * dim, [11] * dim, [100] * dim, [13] * dim],
                     value_dtype), value_dtype)
        exported_exists = constant_op.constant([True, True, False, True],
                                               dtype=dtypes.bool)

        table = de.get_variable(
            'taccum1-' + str(id),
            key_dtype=key_dtype,
            value_dtype=value_dtype,
            initializer=np.array([-1]).astype(_type_converter(value_dtype)),
            dim=dim,
            devices=["/GPU:0"],
            kv_creator=de.HkvHashTableCreator(config=de.HkvHashTableConfig(
                max_capacity=99999)))
        self.evaluate(table.clear())

        self.assertAllEqual(0, self.evaluate(table.size()))

        self.evaluate(table.upsert(base_keys, base_values))
        _, exists = table.lookup(accum_keys, return_exists=True)
        self.assertAllEqual(self.evaluate(exported_exists),
                            self.evaluate(exists))
        # Simulate multi-process situation that other process operated table,
        # between lookup and accum in this process.
        self.evaluate(
            table.upsert(simulate_other_process_add_keys,
                         simulate_other_process_add_vals))
        self.evaluate(table.remove(simulate_other_process_remove_keys))
        self.assertAllEqual(4, self.evaluate(table.size()))
        self.evaluate(
            table.accum(accum_keys, old_values, new_values, exported_exists))

        exported_keys, exported_values = table.export()

        # exported data is in the order of the internal map, i.e. undefined
        sorted_keys = np.sort(self.evaluate(exported_keys), axis=0)
        sorted_values = np.sort(self.evaluate(exported_values), axis=0)
        self.assertAllEqual(
            np.sort(_convert([0, 2, 3, 100], key_dtype), axis=0),
            _convert(sorted_keys, key_dtype))
        self.assertAllEqual(
            _convert([[2] * dim, [10] * dim, [13] * dim, [99] * dim],
                     value_dtype), _convert(sorted_values, value_dtype))

        self.evaluate(table.clear())
        del table

  def test_get_variable(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    with self.session(
        config=default_config,
        graph=ops.Graph(),
        use_gpu=True,
    ):
      default_val = -1
      with variable_scope.variable_scope("embedding", reuse=True):
        table1 = de.get_variable(
            "t1" + '_test_get_variable',
            dtypes.int64,
            dtypes.int32,
            initializer=default_val,
            dim=2,
            devices=["/GPU:0"],
            init_size=2048,
            kv_creator=de.HkvHashTableCreator(config=de.HkvHashTableConfig(
                max_capacity=99999)))
        table2 = de.get_variable(
            "t1" + '_test_get_variable',
            dtypes.int64,
            dtypes.int32,
            initializer=default_val,
            dim=2,
            devices=["/GPU:0"],
            init_size=2048,
            kv_creator=de.HkvHashTableCreator(config=de.HkvHashTableConfig(
                max_capacity=99999)))
        table3 = de.get_variable(
            "t3" + '_test_get_variable',
            dtypes.int64,
            dtypes.int32,
            initializer=default_val,
            dim=2,
            devices=["/GPU:0"],
            init_size=2048,
            kv_creator=de.HkvHashTableCreator(config=de.HkvHashTableConfig(
                max_capacity=99999)))

        table1.clear()
        table2.clear()
        table3.clear()

      self.assertAllEqual(table1, table2)
      self.assertNotEqual(table1, table3)

  def test_get_variable_reuse_error(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    ops.disable_eager_execution()
    with self.session(
        config=default_config,
        graph=ops.Graph(),
        use_gpu=True,
    ):
      with variable_scope.variable_scope("embedding", reuse=False):
        _ = de.get_variable(
            "t900",
            initializer=-1,
            dim=2,
            devices=["/GPU:0"],
            kv_creator=de.HkvHashTableCreator(config=de.HkvHashTableConfig(
                max_capacity=99999)))
        with self.assertRaisesRegexp(ValueError,
                                     "Variable embedding/t900 already exists"):
          _ = de.get_variable(
              "t900",
              initializer=-1,
              dim=2,
              devices=["/GPU:0"],
              kv_creator=de.HkvHashTableCreator(config=de.HkvHashTableConfig(
                  max_capacity=99999)))

  @test_util.run_v1_only("Multiple sessions")
  def test_sharing_between_multi_sessions(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    ops.disable_eager_execution()
    # Start a server to store the table state
    server = server_lib.Server({"local0": ["localhost:0"]},
                               protocol="grpc",
                               start=True)
    # Create two sessions sharing the same state
    session1 = session.Session(server.target, config=default_config)
    session2 = session.Session(server.target, config=default_config)

    table = de.get_variable(
        "tx100" + '_test_sharing_between_multi_sessions',
        dtypes.int64,
        dtypes.int32,
        initializer=0,
        dim=1,
        devices=["/GPU:0"],
        kv_creator=de.HkvHashTableCreator(config=de.HkvHashTableConfig(
            max_capacity=99999)))
    self.evaluate(table.clear())

    # Populate the table in the first session
    with session1:
      with ops.device(_get_devices()[0]):
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(variables.local_variables_initializer())
        self.assertAllEqual(0, table.size().eval())

        keys = constant_op.constant([11, 12], dtypes.int64)
        values = constant_op.constant([[11], [12]], dtypes.int32)
        table.upsert(keys, values).run()
        self.assertAllEqual(2, table.size().eval())

        output = table.lookup(constant_op.constant([11, 12, 13], dtypes.int64))
        self.assertAllEqual([[11], [12], [0]], output.eval())

    # Verify that we can access the shared data from the second session
    with session2:
      with ops.device(_get_devices()[0]):
        self.assertAllEqual(2, table.size().eval())

        output = table.lookup(constant_op.constant([10, 11, 12], dtypes.int64))
        self.assertAllEqual([[0], [11], [12]], output.eval())

  def test_insert_repeat_data(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    id = 0
    dim_list = [1, 2, 4, 8, 10, 16, 32, 64, 100, 200]
    kv_list = [[dtypes.int64, dtypes.int8], [dtypes.int64, dtypes.int32],
               [dtypes.int64, dtypes.int64], [dtypes.int64, dtypes.float32],
               [dtypes.int64, dtypes.half]]
    bits_list = []
    for kv in kv_list:
      bits_list.append(kv[1].size)

    def _convert(v, t):
      return np.array(v).astype(_type_converter(t))

    key_num = 50000
    for (key_dtype,
         value_dtype), dim, bit in itertools.product(kv_list, dim_list,
                                                     bits_list):
      id += 1
      with self.session(config=default_config, use_gpu=True) as sess:
        hkv_table_config = de.HkvHashTableConfig(init_capacity=100000,
                                                 max_capacity=100000,
                                                 max_hbm_for_values=bit *
                                                 (dim + 1) * 100)
        table = de.get_variable(
            't1-' + str(id) + '_test_variable',
            key_dtype=key_dtype,
            value_dtype=value_dtype,
            initializer=np.array([-1]).astype(_type_converter(value_dtype)),
            devices=['/GPU:0'],
            dim=dim,
            kv_creator=de.HkvHashTableCreator(config=hkv_table_config))

        for i in range(2):
          keys = constant_op.constant(
              np.array([x for x in range(key_num)
                       ]).astype(_type_converter(key_dtype)), key_dtype)
          values = constant_op.constant(
              _convert([[x] * dim for x in range(key_num)], value_dtype),
              value_dtype)

          if i == 0:
            self.assertAllEqual(0, self.evaluate(table.size()))
          else:
            self.assertAllEqual(key_num, self.evaluate(table.size()))

          self.evaluate(table.upsert(keys, values))
          self.assertAllEqual(key_num, self.evaluate(table.size()))

        self.evaluate(table.clear())
        del table

  def test_reach_max_hbm(self):
    if not is_gpu_available:
      self.skipTest('Only test when gpu is available.')
    id = 0
    dim_list = [1, 2, 4, 8, 10, 16, 32, 64, 100, 200]
    kv_list = [[dtypes.int64, dtypes.int8], [dtypes.int64, dtypes.int32],
               [dtypes.int64, dtypes.int64], [dtypes.int64, dtypes.float32],
               [dtypes.int64, dtypes.half]]
    bits_list = [1, 4, 8, 4, 2]

    dim_list = [32]
    kv_list = [[dtypes.int64, dtypes.int64]]
    bits_list = [8]

    def _convert(v, t):
      return np.array(v).astype(_type_converter(t))

    k_key_num = 1024 * 1024 * 2
    for (key_dtype,
         value_dtype), dim, bit in itertools.product(kv_list, dim_list,
                                                     bits_list):
      id += 1
      with self.session(config=default_config, use_gpu=True) as sess:
        hkv_table_config = de.HkvHashTableConfig(init_capacity=k_key_num,
                                                 max_capacity=k_key_num,
                                                 max_hbm_for_values=bit *
                                                 (dim + 1) * 1024 * 1024 * 4)
        table = de.get_variable(
            't1-' + str(id) + '_test_variable',
            key_dtype=key_dtype,
            value_dtype=value_dtype,
            initializer=np.array([-1]).astype(_type_converter(value_dtype)),
            dim=dim,
            kv_creator=de.HkvHashTableCreator(config=hkv_table_config))
        table.clear()
        keys = constant_op.constant(
            np.array([x for x in range(0, int(k_key_num / 2))
                     ]).astype(_type_converter(key_dtype)), key_dtype)
        values = constant_op.constant(
            _convert([[x] * dim for x in range(0, int(k_key_num / 2))],
                     value_dtype), value_dtype)

        self.assertAllEqual(0, self.evaluate(table.size()))

        self.evaluate(table.upsert(keys, values))
        self.assertAllEqual(k_key_num / 2, self.evaluate(table.size()))

        keys = constant_op.constant(
            np.array([x for x in range(int(k_key_num / 2), k_key_num)
                     ]).astype(_type_converter(key_dtype)), key_dtype)
        values = constant_op.constant(
            _convert([[x] * dim for x in range(int(k_key_num / 2), k_key_num)],
                     value_dtype), value_dtype)

        self.evaluate(table.upsert(keys, values))
        self.assertAllInRange(self.evaluate(table.size()), k_key_num / 2,
                              k_key_num)

        self.evaluate(table.clear())
        del table


if __name__ == "__main__":
  test.main()
