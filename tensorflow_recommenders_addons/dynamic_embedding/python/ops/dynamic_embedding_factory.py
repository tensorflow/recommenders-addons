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

from tensorflow_recommenders_addons import dynamic_embedding as de

class Singleton(type):
  """
  Meta class for Singleton
  """
  _instances = {}
  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
      cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
    return cls._instances[cls]

class KVcreator(object,metaclass=Singleton):

  def create_instance(
    cls,
    key_dtype=None,
    value_dtype=None,
    default_value=None,
    name=None,
    checkpoint=None,
    init_size=None,
    params_dict=None,
  ):
    raise NotImplementedError('create_instance function must be implemented')

  default_creat_instance_method = create_instance

  def apply_method(cls, creat_instance_method):
    cls.default_creat_instance_method = creat_instance_method

  def create(
    cls, 
    key_dtype=None,
    value_dtype=None,
    default_value=None,
    name=None,
    checkpoint=None,
    init_size=0,
    params_dict=None,
  ):
    return cls.default_creat_instance_method(
              key_dtype=key_dtype,
              value_dtype=value_dtype,
              default_value=default_value,
              name=name,
              checkpoint=checkpoint,
              init_size=init_size,
              params_dict=params_dict,
            )
  
class CuckooHashTableCreator(KVcreator):
  def create_instance(
    self,
    key_dtype=None,
    value_dtype=None,
    default_value=None,
    name=None,
    checkpoint=None,
    init_size=0,
    params_dict={},
  ):
    return de.CuckooHashTable(
              key_dtype=key_dtype,
              value_dtype=value_dtype,
              default_value=default_value,
              name=name,
              checkpoint=checkpoint,
              init_size=init_size,
              params_dict=params_dict,
          )

  def __init__(self):
    _KVcreator = KVcreator()
    _KVcreator.apply_method(self.create_instance)

class RedisTableCreator(KVcreator):
  def create_instance(
    self,
    key_dtype=None,
    value_dtype=None,
    default_value=None,
    name=None,
    checkpoint=None,
    init_size=0,
    params_dict={},
  ):
    return de.RedisTable(
              key_dtype=key_dtype,
              value_dtype=value_dtype,
              default_value=default_value,
              name=name,
              checkpoint=checkpoint,
              init_size=init_size,
              params_dict=params_dict,
          )

  def __init__(self):
    _KVcreator = KVcreator()
    _KVcreator.apply_method(self.create_instance)

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
    