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
from tensorflow_recommenders_addons import dynamic_embedding as de

class KVCreator(object,metaclass=ABCMeta):
  def __init__(self, config=None):
    self.config = config

  def create(self,
              key_dtype=None,
              value_dtype=None,
              default_value=None,
              name=None,
              checkpoint=None,
              init_size=None,
              config=None):

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
      init_size=None,
      config=None,
  ):
    return de.CuckooHashTable(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        default_value=default_value,
        name=name,
        checkpoint=checkpoint,
        init_size=init_size,
        config=config,
    )


class RedisTableConfig(object):
  def __init__(
    self,
    redis_connection_mode=1, # ClusterMode = 0, SentinelMode = 1, StreamMode = 2
    redis_master_name="master",
    # connection_options
    redis_host_ip="127.0.0.1",
    redis_host_port=26379,
    redis_password="",
    redis_db=0,
    redis_connect_timeout=1000, # milliseconds
    redis_socket_timeout=1000,  # milliseconds
    # connection_pool_options
    redis_conn_pool_size=20,
    redis_wait_timeout=100000000,  # milliseconds
    redis_connection_lifetime=100, # minutes
    # sentinel_connection_options
    redis_sentinel_connect_timeout=1000, # milliseconds
    sentinel_socket_timeout=1000,  # milliseconds
    # Below there is user-defined parameters in this custom op, not Redis setting parameters
    storage_slice=1, # For deciding hash tag, which usually is how many Redis instance may be used in the trainning. For performance reasons, it is recommended that each slice be no larger than 256MB.
    using_MD5_prefix_name=False, # 1=true, 0=false
    model_tag="test", #  model_tag for version and any other information
    using_model_lib=True,
    model_lib_abs_dir="/tmp/",
  ):
    self.redis_connection_mode=redis_connection_mode # ClusterMode = 0, SentinelMode = 1, StreamMode = 2
    self.redis_master_name=redis_master_name
    # connection_options
    self.redis_host_ip=redis_host_ip
    self.redis_host_port=redis_host_port
    self.redis_password=redis_password
    self.redis_db=redis_db
    self.redis_connect_timeout=redis_connect_timeout # milliseconds
    self.redis_socket_timeout=redis_socket_timeout # milliseconds
    # connection_pool_options
    self.redis_conn_pool_size=redis_conn_pool_size
    self.redis_wait_timeout=redis_wait_timeout # milliseconds
    self.redis_connection_lifetime=redis_connection_lifetime # minutes
    # sentinel_connection_options
    self.redis_sentinel_connect_timeout=redis_sentinel_connect_timeout # milliseconds
    self.sentinel_socket_timeout=sentinel_socket_timeout # milliseconds
    # Below there is user-defined parameters in this custom op, not Redis setting parameters
    self.storage_slice=storage_slice # For deciding hash tag, which usually is how many Redis instance may be used in the trainning. For performance reasons, it is recommended that each slice be no larger than 256MB.
    self.using_MD5_prefix_name=using_MD5_prefix_name # 1=true, 0=false
    self.model_tag=model_tag #  model_tag for version and any other information
    self.using_model_lib=using_model_lib
    self.model_lib_abs_dir=model_lib_abs_dir


class RedisTableCreator(KVCreator):
  def create(
      self,
      key_dtype=None,
      value_dtype=None,
      default_value=None,
      name=None,
      checkpoint=None,
      init_size=None,
      config=None,
  ):
    real_config = config if config is not None else self.config
    if not isinstance(real_config, RedisTableConfig):
      raise TypeError("config should be instance of 'config', but got ",
                      str(type(real_config)))
    return de.RedisTable(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        default_value=default_value,
        name=name,
        checkpoint=checkpoint,
        config=self.config,
    )

# class KVcreatorMeta(type):
#   '''It's not useless at the moment'''
#   def __init__(self, name, bases, dic):
#     super().__init__(name, bases, dic)

#   def __new__(cls, *args, **kwargs):
#     return type.__new__(cls, *args, **kwargs)

#   def __call__(cls, *args, **kwargs):
#     obj = cls.__new__(cls)
#     cls.__init__(cls, *args, **kwargs)
#     return obj
    
# class Singleton(type):
#   """
#   Meta class for Singleton
#   """
#   _instances = {}
#   def __call__(cls, *args, **kwargs):
#     if cls not in cls._instances:
#       cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
#     return cls._instances[cls]