# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Additional layers for sequence to sequence models."""

from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_ops import create_slots
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_ops import embedding_lookup
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_ops import embedding_lookup_sparse
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_ops import get_variable
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_ops import safe_embedding_lookup_sparse
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_ops import Variable
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_optimizer import DynamicEmbeddingOptimizer
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.tf_hacker import hacking_tf
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.trainable_wrapper import TrainableWrapper

hacking_tf()
