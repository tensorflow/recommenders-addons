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
"""Export dynamic_embedding APIs."""

__all__ = [
    'CuckooHashTable',
    'CuckooHashTableConfig',
    'CuckooHashTableCreator',
    'HkvEvictStrategy',
    'HkvHashTable',
    'HkvHashTableConfig',
    'HkvHashTableCreator',
    'RedisTable',
    'RedisTableConfig',
    'RedisTableCreator',
    'Variable',
    'TrainableWrapper',
    'DistributedVariableWrapper',
    'DynamicEmbeddingOptimizer',
    'GraphKeys',
    'ModelMode',
    'RestrictPolicy',
    'TimestampRestrictPolicy',
    'FrequencyRestrictPolicy',
    'get_variable',
    'embedding_lookup',
    'embedding_lookup_sparse',
    'embedding_lookup_unique',
    'safe_embedding_lookup_sparse',
    'enable_inference_mode',
    'enable_train_mode',
    'get_model_mode',
    'trainable_wrapper_filter',
    'train',
    'keras',
    'math',
    'data_flow',
    'shadow_ops',
    'DEResourceVariable',
]

from tensorflow_recommenders_addons.dynamic_embedding.python import train
from tensorflow_recommenders_addons.dynamic_embedding.python import keras
from tensorflow_recommenders_addons.dynamic_embedding.python.ops import math_ops as math
from tensorflow_recommenders_addons.dynamic_embedding.python.ops import data_flow_ops as data_flow
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_creator import (
    KVCreator, CuckooHashTableConfig, CuckooHashTableCreator,
    HkvHashTableConfig, HkvHashTableCreator, HkvEvictStrategy, RedisTableConfig,
    RedisTableCreator, FileSystemSaver)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.cuckoo_hashtable_ops import (
    CuckooHashTable,)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.hkv_hashtable_ops import (
    HkvHashTable,)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.redis_table_ops import (
    RedisTable,)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_variable import (
    embedding_lookup,)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_ops import (
    embedding_lookup_sparse,)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_ops import (
    embedding_lookup_unique,)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_ops import (
    safe_embedding_lookup_sparse,)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_ops import (
    enable_inference_mode,)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_ops import (
    enable_train_mode,)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_ops import (
    get_model_mode,)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_ops import (
    trainable_wrapper_filter,)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.embedding_weights import (
    ModelMode,
    TrainableWrapper,
)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.shadow_embedding_ops import DEResourceVariable
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.distributed_embedding_variable import \
    DistributedVariableWrapper
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_optimizer import (
    create_slots,)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_optimizer import (
    DynamicEmbeddingOptimizer,)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_variable import (
    get_variable,
    embedding_lookup,
)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_variable import (
    Variable,)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_variable import (
    GraphKeys,)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.warm_start_util import (
    warm_start, WarmStartHook)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.restrict_policies import (
    RestrictPolicy,
    TimestampRestrictPolicy,
    FrequencyRestrictPolicy,
)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.tf_patch import (
    patch_on_tf,)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.tf_save_restore_patch import (
    patch_on_tf_save_restore,)
from tensorflow_recommenders_addons.dynamic_embedding.python.ops import shadow_embedding_ops as shadow_ops
from tensorflow_recommenders_addons.dynamic_embedding.python.ops import ragged_embedding_ops

patch_on_tf()
patch_on_tf_save_restore()
