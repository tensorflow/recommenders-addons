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

class TableFactory:
  '''
  Use table_fn to create the corresponding sparse table.
  '''
  sparse_table = {
    'CuckooHashTable': de.CuckooHashTable, 
    'Redis': None
    }
    
  def __new__(
    cls, 
    table_fn=None,
    key_dtype=None,
    value_dtype=None,
    default_value=None,
    name=None,
    checkpoint=None,
    init_size=None,
    ):
    if table_fn in cls.sparse_table:
      return cls.sparse_table[table_fn](
                key_dtype=key_dtype,
                value_dtype=value_dtype,
                default_value=default_value,
                name=name,
                checkpoint=checkpoint,
                init_size=init_size,
            )
    else:
      raise ValueError("There is no sparse table creator called ",table_fn)

class TableFactoryMeta(type):
  '''It's not useless at the moment'''
  def __init__(self, name, bases, dic):
    super().__init__(name, bases, dic)

  def __new__(cls, *args, **kwargs):
    return type.__new__(cls, *args, **kwargs)

  def __call__(cls, *args, **kwargs):
    obj = cls.__new__(cls)
    cls.__init__(cls, *args, **kwargs)
    return obj