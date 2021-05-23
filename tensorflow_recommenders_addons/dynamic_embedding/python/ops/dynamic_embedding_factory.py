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
  '''
  Use table_fn to create the corresponding sparse table.
  '''
  sparse_table = {
    'CuckooHashTable': de.CuckooHashTable, 
    'RedisTable': de.RedisTable,
    }
  
  table_fn_instance='CuckooHashTable'
  table_params_dict={}

  def __new__(
    cls,
    *args,
  ):
    if(len(args)==2):
      if(isinstance(args[0],str)):
        cls.table_fn_instance=args[0]
        cls.table_params_dict=args[1]
      elif(isinstance(args[1],str)):
        cls.table_fn_instance=args[1]
        cls.table_params_dict=args[0]
    else:
      raise NotImplementedError("Please provide a string name for the table and a dictionary variable for the line of sight argument of the corresponding table")

  @classmethod
  def instance(
    cls, 
    key_dtype=None,
    value_dtype=None,
    default_value=None,
    name=None,
    checkpoint=None,
    init_size=None,
    ):
    if cls.table_fn_instance in cls.sparse_table:
      return cls.sparse_table[cls.table_fn_instance](
                key_dtype=key_dtype,
                value_dtype=value_dtype,
                default_value=default_value,
                name=name,
                checkpoint=checkpoint,
                init_size=init_size,
                params=cls.table_params_dict,
            )
    else:
      raise NotImplementedError("There is no sparse table creator called ",cls.table_fn_instance)





class KVcreatorMeta(type):
  '''It's not useless at the moment'''
  def __init__(self, name, bases, dic):
    super().__init__(name, bases, dic)

  def __new__(cls, *args, **kwargs):
    return type.__new__(cls, *args, **kwargs)

  def __call__(cls, *args, **kwargs):
    obj = cls.__new__(cls)
    cls.__init__(cls, *args, **kwargs)
    return obj
    