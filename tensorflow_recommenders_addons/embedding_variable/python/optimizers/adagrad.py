# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""AdagradOptimizer for TensorFlow Recommenders Addons."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import math_ops
from tensorflow.python.training import adagrad
from tensorflow.python.training import training_ops
from tensorflow.python.training import training_util
from tensorflow_recommenders_addons.embedding_variable.python import gen_ev_ops
from tensorflow_recommenders_addons.embedding_variable.python.ops import embedding_variable_ops


class AdagradOptimizer(adagrad.AdagradOptimizer):

  def _resource_apply_sparse(self, grad, var, indices):
    acc = self.get_slot(var, "accumulator")
    if isinstance(var, embedding_variable_ops.EmbeddingVariable):
      global_step = training_util.get_or_create_global_step()
      return gen_ev_ops.ev_sparse_apply_adagrad(var.handle,
                                                acc.handle,
                                                math_ops.cast(
                                                    self._learning_rate_tensor,
                                                    grad.dtype),
                                                grad,
                                                indices,
                                                global_step,
                                                use_locking=self._use_locking)
    else:
      return training_ops.resource_sparse_apply_adagrad(
          var.handle,
          acc.handle,
          math_ops.cast(self._learning_rate_tensor, grad.dtype),
          grad,
          indices,
          use_locking=self._use_locking)
