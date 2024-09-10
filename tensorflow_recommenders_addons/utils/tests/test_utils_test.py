import random
try:
  from packaging.version import Version
except:  # make it compatible for python 3.7
  from distutils.version import LooseVersion as Version

import numpy as np
import pytest
import tensorflow as tf
from tensorflow_recommenders_addons.utils import test_utils

try:
  from tf_keras import layers, Model
except:
  from tensorflow.keras import layers, Model


def test_seed_is_set():
  assert random.randint(0, 10000) == 6311
  assert np.random.randint(0, 10000) == 2732
  tf_random = tf.random.uniform([], 0, 10000, dtype=tf.int64)
  if hasattr(tf_random, "numpy"):
    assert tf_random.numpy() == 9457


@pytest.mark.with_device(["cpu", "gpu", tf.distribute.MirroredStrategy])
def test_all_scopes(device):
  assert isinstance(device, str) or isinstance(device, tf.distribute.Strategy)


def train_small_model():
  model_input = layers.Input((3,))
  model_output = layers.Dense(4)(model_input)
  model = Model(model_input, model_output)
  model.compile(loss="mse")

  x = np.random.uniform(size=(5, 3))
  y = np.random.uniform(size=(5, 4))
  model.fit(x, y, epochs=1)


@pytest.mark.skipif(
    Version(tf.__version__) >= Version("2.13"),
    reason=
    "TF2.13 breakage: https://github.com/tensorflow/addons/pull/2835#issuecomment-1629772331",
)
@pytest.mark.with_device([tf.distribute.MirroredStrategy])
def test_distributed_strategy(device):
  assert isinstance(device, tf.distribute.Strategy)
  train_small_model()


@pytest.mark.skipif(
    Version(tf.__version__) >= Version("2.13"),
    reason=
    "TF2.13 breakage: https://github.com/tensorflow/addons/pull/2835#issuecomment-1629772331",
)
@pytest.mark.with_device(["no_device"])
@pytest.mark.needs_gpu
def test_custom_device_placement():
  with tf.device(test_utils.gpus_for_testing()[0]):
    train_small_model()

  strategy = tf.distribute.MirroredStrategy(test_utils.gpus_for_testing())
  with strategy.scope():
    train_small_model()
