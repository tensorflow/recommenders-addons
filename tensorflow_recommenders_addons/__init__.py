# Copyright 2020 The TensorFlow Recommenders Addons Authors.
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
"""TensorFlow Recommenders Addon is a library for building recommender system models.

It helps with the full workflow of building a recommender system: data
preparation, model formulation, training, evaluation, and deployment.

It's built on Keras and aims to have a gentle learning curve while still giving
you the flexibility to build complex models.
"""
__all__ = ['dynamic_embedding', 'embedding_variable']

from tensorflow_recommenders_addons.utils.ensure_tf_install import _check_tf_version
from tensorflow_recommenders_addons.version import __version__

_check_tf_version()

from tensorflow_recommenders_addons import dynamic_embedding
from tensorflow_recommenders_addons import embedding_variable
from tensorflow_recommenders_addons.register import register_all
