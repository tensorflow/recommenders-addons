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
"""EmbeddingVariable."""

import os
import ctypes
import sys
import inspect

import tensorflow as tf


def _load_library(filename, lib="op"):
  """_load_library"""
  f = inspect.getfile(sys._getframe(1))  # pylint: disable=protected-access

  # Construct filename
  f = os.path.join(os.path.dirname(f), filename)
  filenames = [f]

  # Function to load the library, return True if file system library is loaded
  if lib == "op":
    load_fn = tf.load_op_library
  elif lib == "dependency":
    load_fn = lambda f: ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
  else:
    load_fn = lambda f: tf.load_library(f) is None

  # Try to load all paths for file, fail if none succeed
  errs = []
  for f in filenames:
    try:
      l = load_fn(f)
      if l is not None:
        return l
    except (tf.errors.NotFoundError, OSError) as e:
      errs.append(str(e))
  raise NotImplementedError(
      "unable to open file: " +
      "{}, from paths: {}\ncaused by: {}".format(filename, filenames, errs))


gen_ev_ops = _load_library("../core/_ev_ops.so")
