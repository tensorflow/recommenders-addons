# Copyright 2020 The TensorFlow Recommenders-Addons Authors.
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

# lint-as: python3

from abc import ABCMeta
from enum import IntEnum, unique

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow_recommenders_addons import dynamic_embedding as de

KHkvHashTableInitCapacity = 1024 * 1024
KHkvHashTableMaxCapacity = 1024 * 1024
KHkvHashTableMaxHbmForValuesByBytes = 1024 * 1024 * 1024


class KVCreator(object, metaclass=ABCMeta):
  """
      A generic KV table creator.

      KV table instance will be created by the create function with config.
    And also a config class for specific table instance backend should be
    inited before callling the creator function.
      And then, the KVCreator class instance will be passed to the Variable
    class for creating the real KV table backend(TF resource).

    Example usage:

    ```python
    redis_config1=tfra.dynamic_embedding.RedisTableConfig(
      redis_config_abs_dir="xx/yy.json"
    )
    redis_creator1=tfra.dynamic_embedding.RedisTableCreator(
      config=redis_config1,
      saver=FileSystemSaverConfig()
    )
    ```
  """

  def __init__(self, config=None, saver=None):
    self.config = config
    self.saver = saver
    if saver and not isinstance(saver, DynamicEmbeddingSaver):
      raise RuntimeError(
          'The initialization argument \'saver\' for class KVCreator must be a class inheriting DynamicEmbeddingSaver.'
      )

  def create(self,
             key_dtype=None,
             value_dtype=None,
             default_value=None,
             name=None,
             checkpoint=None,
             init_size=None,
             config=None,
             device=None):

    raise NotImplementedError('create function must be implemented')


class CuckooHashTableConfig(object):

  def __init__(self):
    """ CuckooHashTableConfig include nothing for parameter default satisfied.
    """
    pass


class CuckooHashTableCreator(KVCreator):

  def create(
      self,
      key_dtype=None,
      value_dtype=None,
      default_value=None,
      name=None,
      checkpoint=None,
      init_size=0,
      config=None,
      device=None,
      shard_saveable_object_fn=None,
  ):
    self.key_dtype = key_dtype
    self.value_dtype = value_dtype
    self.default_value = default_value
    self.name = name
    self.checkpoint = checkpoint
    self.init_size = init_size
    self.config = config
    self.device = device
    self.shard_saveable_object_fn = shard_saveable_object_fn
    return de.CuckooHashTable(
        key_dtype=self.key_dtype,
        value_dtype=self.value_dtype,
        default_value=self.default_value,
        name=self.name,
        checkpoint=self.checkpoint,
        init_size=self.init_size,
        config=self.config,
        device=self.device,
        shard_saveable_object_fn=self.shard_saveable_object_fn)

  def get_config(self):
    if not context.executing_eagerly():
      raise RuntimeError(
          'Unsupported to serialize python object of CuckooHashTableCreator.')

    config = {
        'key_dtype': self.key_dtype,
        'value_dtype': self.value_dtype,
        'default_value': self.default_value.numpy(),
        'name': self.name,
        'checkpoint': self.checkpoint,
        'init_size': self.init_size,
        'config': self.config,
        'device': self.device,
    }
    return config


@unique
class HkvEvictStrategy(IntEnum):
  LRU = 0
  LFU = 1
  EPOCHLRU = 2
  EPOCHLFU = 3
  CUSTOMIZED = 4


class HkvHashTableConfig(object):

  def __init__(
      self,
      init_capacity=KHkvHashTableInitCapacity,
      max_capacity=KHkvHashTableMaxCapacity,
      max_hbm_for_values=KHkvHashTableMaxHbmForValuesByBytes,
      evict_strategy=HkvEvictStrategy.LRU,
      step_per_epoch=0,
      gen_scores_fn=None,
      reserved_key_start_bit=0,
  ):
    """ CuckooHashTableConfig include nothing for parameter default satisfied.
    """
    self.init_capacity = init_capacity
    self.max_capacity = max_capacity
    self.max_hbm_for_values = max_hbm_for_values
    self.evict_strategy = evict_strategy
    self.step_per_epoch = step_per_epoch
    self.gen_scores_fn = gen_scores_fn
    self.reserved_key_start_bit = reserved_key_start_bit


class HkvHashTableCreator(KVCreator):

  def create(
      self,
      key_dtype=None,
      value_dtype=None,
      default_value=None,
      name=None,
      checkpoint=None,
      init_size=KHkvHashTableInitCapacity,
      config=None,
      device=None,
      shard_saveable_object_fn=None,
  ):
    self.key_dtype = key_dtype
    self.value_dtype = value_dtype
    self.default_value = default_value
    self.name = name
    self.checkpoint = checkpoint
    self.init_capacity = init_size
    self.max_capacity = KHkvHashTableMaxCapacity
    self.max_hbm_for_values = KHkvHashTableMaxHbmForValuesByBytes
    self.evict_strategy = HkvEvictStrategy.LRU
    self.step_per_epoch = 0
    self.gen_scores_fn = None
    self.reserved_key_start_bit = 0
    if self.config and isinstance(self.config, de.HkvHashTableConfig):
      self.init_capacity = self.config.init_capacity
      self.max_capacity = self.config.max_capacity
      self.max_hbm_for_values = self.config.max_hbm_for_values
      self.evict_strategy = self.config.evict_strategy
      self.step_per_epoch = self.config.step_per_epoch
      self.gen_scores_fn = self.config.gen_scores_fn
      self.reserved_key_start_bit = self.config.reserved_key_start_bit
    self.device = device
    self.shard_saveable_object_fn = shard_saveable_object_fn

    return de.HkvHashTable(
        key_dtype=self.key_dtype,
        value_dtype=self.value_dtype,
        default_value=self.default_value,
        name=self.name,
        checkpoint=self.checkpoint,
        init_capacity=self.init_capacity,
        max_capacity=self.max_capacity,
        max_hbm_for_values=self.max_hbm_for_values,
        evict_strategy=self.evict_strategy,
        step_per_epoch=self.step_per_epoch,
        gen_scores_fn=self.gen_scores_fn,
        config=self.config,
        device=self.device,
        shard_saveable_object_fn=self.shard_saveable_object_fn,
        reserved_key_start_bit=self.reserved_key_start_bit)

  def get_config(self):
    if not context.executing_eagerly():
      raise RuntimeError(
          'Unsupported to serialize python object of HkvHashTableCreator.')

    config = {
        'key_dtype': self.key_dtype,
        'value_dtype': self.value_dtype,
        'default_value': self.default_value.numpy(),
        'name': self.name,
        'checkpoint': self.checkpoint,
        'init_capacity': self.init_capacity,
        'max_capacity': self.max_capacity,
        'config': self.config,
        'device': self.device,
    }
    return config


class RedisTableConfig(object):
  """ 
  RedisTableConfig config json file for connecting Redis service and 
  assign the embedding table starage properties.
  An example of a configuration file is shown below:
  ```python
  {
    "redis_connection_mode": 2,
    "redis_master_name": "master",
    "redis_host_ip": [
      "127.0.0.1"
    ],
    "redis_host_port": 6379,
    "redis_user": "default",
    "redis_password": "",
    "redis_db": 0,
    "redis_read_access_slave": False,
    "redis_connect_keep_alive": False,
    "redis_connect_timeout": 1000,
    "redis_socket_timeout": 1000,
    "redis_conn_pool_size": 20,
    "redis_wait_timeout": 100000000,
    "redis_connection_lifetime": 100,
    "redis_sentinel_user": "default",
    "redis_sentinel_password": "",
    "redis_sentinel_connect_timeout": 1000,
    "redis_sentinel_socket_timeout": 1000,
    "storage_slice_import": 1,
    "storage_slice": 1,
    "using_hash_storage_slice": False,
    "keys_sending_size": 1024,
    "using_md5_prefix_name": False,
    "redis_hash_tags_hypodispersion": False,
    "model_tag_import": "test",
    "redis_hash_tags_import": [],
    "model_tag_runtime": "test",
    "redis_hash_tags_runtime": [],
    "expire_model_tag_in_seconds": 604800,
    "table_store_mode": 1,
    "model_lib_abs_dir": "/tmp/"
  }
  ```
  Refer to the [Redis table config guide](https://github.com/tensorflow/recommenders-addons/blob/master/docs/api_docs/tfra/dynamic_embedding/RedisBackend.md)
  and default_redis_params variable in RedisTable class 
  to learn the description of the JSON configuration file
  """

  def __init__(
      self,
      redis_config_abs_dir=None,
      redis_config_abs_dir_env=None,
  ):
    self.redis_config_abs_dir = redis_config_abs_dir
    self.redis_config_abs_dir_env = redis_config_abs_dir_env


class RedisTableCreator(KVCreator):
  """ 
      RedisTableCreator will create a object to pass itself to the others classes
    for creating a real Redis client instance which can interact with TF.
  """

  def create(
      self,
      key_dtype=None,
      value_dtype=None,
      default_value=None,
      name=None,
      checkpoint=None,
      init_size=None,
      config=None,
      device=None,
      shard_saveable_object_fn=None,
  ):
    self.key_dtype = key_dtype
    self.value_dtype = value_dtype
    self.default_value = default_value
    self.name = name
    self.checkpoint = checkpoint
    self.init_size = init_size
    if config:
      if not isinstance(config, RedisTableConfig):
        raise TypeError(
            "config should be instance of 'RedisTableConfig', but got ",
            str(type(config)))
      self.config = config
    else:
      if self.config is None:
        self.config = de.RedisTableConfig(None, None)
    self.device = device
    self.shard_saveable_object_fn = shard_saveable_object_fn

    return de.RedisTable(key_dtype=self.key_dtype,
                         value_dtype=self.value_dtype,
                         default_value=self.default_value,
                         name=self.name,
                         checkpoint=self.checkpoint,
                         config=self.config,
                         device=self.device,
                         shard_saveable_object_fn=self.shard_saveable_object_fn)

  def get_config(self):
    if not context.executing_eagerly():
      raise RuntimeError(
          'Unsupported to serialize python object of RedisTableCreator.')

    config = {
        'key_dtype': self.key_dtype,
        'value_dtype': self.value_dtype,
        'default_value': self.default_value.numpy(),
        'name': self.name,
        'checkpoint': self.checkpoint,
        'init_size': self.init_size,
        'config': self.config,
        'device': self.device,
    }
    return config


# DynamicEmbeddingSaver
class DynamicEmbeddingSaver(object, metaclass=ABCMeta):
  """
    Example usage:
      ```python
      kv_creator = tfra.dynamic_embedding.CuckooHashTableCreator(saver=FileSystemSaver())
      ```
  """
  _upsert_restore = True

  def __init__(self):
    raise NotImplementedError('create function must be implemented')

  def create_variable_saveable_object(self, variable, name):
    raise NotImplementedError('create function must be implemented')

  def create_shard_saveable_object(self,
                                   variable,
                                   table,
                                   shard_idx,
                                   name,
                                   full_name=""):
    raise NotImplementedError('create function must be implemented')

  def set_upsert_restore(self, setting):
    self._upsert_restore = setting


class FileSystemSaverConfig(object):

  def __init__(self,
               proc_size: int = None,
               proc_rank: int = None,
               save_path: str = None,
               buffer_size: int = 4096):
    """ FileSystemSaverConfig can be used to assign save_path of DynamicEmbeddings.
    """
    if type(proc_rank) != type(proc_size):
      raise TypeError(
          "proc_rank and proc_size in FileSystemSaverConfig must both be set to integer properly!"
      )
    if proc_size is None or proc_rank is None:
      self.proc_size = 1
      self.proc_rank = 0
    else:
      self.proc_size = proc_size
      self.proc_rank = proc_rank
    self.save_path = save_path
    self.buffer_size = buffer_size


class FileSystemSaver(DynamicEmbeddingSaver):
  """
    A saver for easily saving/restoring independent KV files for DynamicEmbedding when using TensorFlow savedmodel/checkpoint API.
    
    Args:
      proc_size: A int parameter to config the size of global running TensorFlow instance(process), which usually set during MPI training.
      proc_rank: A int parameter to config the rank of local runtime TensorFlow instance(process), which usually set during MPI training.
      save_path: A string parameter for specific KV files saving path.
      buffer_size: A int parameter to set writing/reading how many keys at a time when handle KV files.

    Example usage:
      ```python
      kv_creator = tfra.dynamic_embedding.CuckooHashTableCreator(saver=FileSystemSaver())
      ```
  """

  class _DynamicEmbeddingVariabelFileSystemSaveable(
      BaseSaverBuilder.SaveableObject):
    """SaveableObject implementation for dynamic embedding."""

    def __init__(self, de_variable, saver_config, name):
      self._de_variable = de_variable
      self.device_size = len(de_variable.devices)
      self.local_shard_num = self.device_size
      self.proc_size = saver_config.proc_size
      self.global_shard_num = self.device_size * self.proc_size
      self.proc_rank = saver_config.proc_rank
      self._saver_config = saver_config
      specs = [
          BaseSaverBuilder.SaveSpec(
              constant_op.constant(self.global_shard_num, dtype=dtypes.int32),
              "", name + "-prev_global_shard_num"),
      ]
      # pylint: disable=protected-access
      super(FileSystemSaver._DynamicEmbeddingVariabelFileSystemSaveable,
            self).__init__(de_variable, specs, name)
      self._restore_name = de_variable.name

    def restore(self, restored_tensors, restored_shapes, name=None):
      self._de_variable.prev_global_shard_num = restored_tensors[0]
      return control_flow_ops.no_op()

  class _DynamicEmbeddingShardFileSystemSaveable(BaseSaverBuilder.SaveableObject
                                                ):
    """SaveableObject implementation for shards of table."""

    def __init__(self,
                 de_variable,
                 shard,
                 shard_idx,
                 shard_upsert_restore,
                 saver_config,
                 name,
                 full_name=''):
      self._de_variable = de_variable
      self.local_shard_idx = shard_idx
      self.device_size = len(de_variable.devices)
      self.local_shard_num = self.device_size
      self.proc_size = saver_config.proc_size
      self.global_shard_num = self.device_size * self.proc_size
      self.proc_rank = saver_config.proc_rank
      self.global_shard_idx = self.proc_rank * self.device_size + shard_idx
      self._shard_upsert_restore = shard_upsert_restore
      self._saver_config = saver_config
      specs = [
          BaseSaverBuilder.SaveSpec(
              constant_op.constant(shard_upsert_restore, dtypes.bool), "",
              name + "-shard_upsert_restore"),
      ]
      # pylint: disable=protected-access
      super(FileSystemSaver._DynamicEmbeddingShardFileSystemSaveable,
            self).__init__(shard, specs, name)
      self._restore_name = shard.name

    def restore(self, restored_tensors, restored_shapes, name=None):
      return control_flow_ops.no_op()

  def __init__(self,
               proc_size: int = None,
               proc_rank: int = None,
               save_path: str = None,
               buffer_size: int = 4096):
    self.config = FileSystemSaverConfig(proc_size=proc_size,
                                        proc_rank=proc_rank,
                                        save_path=save_path,
                                        buffer_size=buffer_size)

  def create_variable_saveable_object(self, variable, name):
    _variable_saveable_obj = self._DynamicEmbeddingVariabelFileSystemSaveable(
        variable, self.config, name)
    return _variable_saveable_obj

  def create_shard_saveable_object(self,
                                   variable,
                                   table,
                                   shard_idx,
                                   name,
                                   full_name=""):
    _shard_saveable_obj = self._DynamicEmbeddingShardFileSystemSaveable(
        variable, table, shard_idx, self._upsert_restore, self.config, name,
        full_name)
    return _shard_saveable_obj
