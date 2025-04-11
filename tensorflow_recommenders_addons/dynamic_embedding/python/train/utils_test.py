import unittest

from tensorflow.python.distribute import multi_worker_test_base

from tensorflow_recommenders_addons.dynamic_embedding.python.train.utils import worker_devices, \
  is_parameter_server_strategy
from tensorflow import distribute as tf_dist
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.training import server_lib
from tensorflow.python.distribute import multi_process_runner


class TestWorkerDevices(unittest.TestCase):

  def test_valid_cases(self):
    self.assertEqual(
        worker_devices(["/device:CPU:0", "/device:CPU:1"], 4, "worker"), [
            "/job:worker/task:0/device:CPU:0",
            "/job:worker/task:1/device:CPU:1",
            "/job:worker/task:2/device:CPU:0",
            "/job:worker/task:3/device:CPU:1",
        ])
    self.assertEqual(worker_devices(["/device:GPU:0"], 2, "worker"), [
        "/job:worker/task:0/device:GPU:0",
        "/job:worker/task:1/device:GPU:0",
    ])

  def test_invalid_cases(self):
    with self.assertRaises(ValueError):
      worker_devices(["/device:CPU:0", "/device:CPU:1"], 3, "worker")

  def test_ps_strategy(self):
    # create ParameterServerStrategy needs bzl test, so we just test the False Case
    s2 = tf_dist.MirroredStrategy()
    self.assertFalse(is_parameter_server_strategy(s2))
    self.assertFalse(is_parameter_server_strategy(None))
