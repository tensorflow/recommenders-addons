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
"""AdamOptimizer for TensorFlow Recommenders Addons."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import math_ops
from tensorflow.python.training import adam
from tensorflow.python.training import training_ops
from tensorflow.python.training import training_util
from tensorflow_recommenders_addons.embedding_variable.python import gen_ev_ops
from tensorflow_recommenders_addons.embedding_variable.python.ops import embedding_variable_ops


class AdamOptimizer(adam.AdamOptimizer):

  def _resource_apply_sparse(self, grad, var, indices):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta1_power, beta2_power = self._get_beta_accumulators()
    if isinstance(var, embedding_variable_ops.EmbeddingVariable):
      global_step = training_util.get_or_create_global_step()
      return gen_ev_ops.ev_sparse_apply_adam(
          var.handle,
          m.handle,
          v.handle,
          beta1_power.read_value(),
          beta2_power.read_value(),
          math_ops.cast(self._lr_t, grad.dtype),
          math_ops.cast(self._beta1_t, grad.dtype),
          math_ops.cast(self._beta2_t, grad.dtype),
          math_ops.cast(self._epsilon_t, grad.dtype),
          grad,
          indices,
          global_step,
          use_locking=self._use_locking)
    else:
      return self._apply_sparse_shared(grad, var, indices,
                                       self._resource_scatter_add)
