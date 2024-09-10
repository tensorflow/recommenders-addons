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
"""Types for typing functions signatures."""
# pylint: disable=protected-access

from typing import Union, Callable, List

import numpy as np
import tensorflow as tf

try:  # tf version >= 2.16
  from tf_keras.optimizers import Optimizer as keras_OptimizerV2
except:
  # Keras version >= 2.12.0
  from tensorflow.keras.optimizers import Optimizer as keras_OptimizerV2

Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

Initializer = Union[None, dict, str, Callable]
Regularizer = Union[None, dict, str, Callable]
Constraint = Union[None, dict, str, Callable]
Activation = Union[None, str, Callable]
Optimizer = Union[keras_OptimizerV2, str]

TensorLike = Union[
    List[Union[Number, list]],
    tuple,
    Number,
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable,
]
FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
AcceptableDTypes = Union[tf.DType, np.dtype, type, int, str, None]
# pylint: enable=protected-access
