# pytest: skip
import os
import sys

from tensorflow.python.distribute import multi_process_lib
import multiprocessing
import tensorflow as tf
from tensorflow.python.framework import constant_op

from tensorflow.python.training import server_lib

from tensorflow_recommenders_addons import dynamic_embedding as de

import numpy as np
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib

from tensorflow.python.eager import test
from packaging import version
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.eager import def_function
from tensorflow.python.ops import variables

if version.parse(tf.__version__) >= version.parse("2.16"):
  from tf_keras import layers
  from tf_keras import Sequential
  from tf_keras.optimizers import Adam
else:
  from tensorflow.python.keras import layers
  from tensorflow.python.keras import Sequential
  try:
    from tensorflow.keras.optimizers import Adam
  except:
    from tensorflow.keras.optimizers.legacy import Adam


def create_multi_process_cluster(cluster_spec,
                                 rpc_layer='grpc',
                                 stream_output=False,
                                 collective_leader=None):

  cluster = multi_worker_test_base.MultiProcessCluster(
      cluster_resolver_lib.SimpleClusterResolver(
          server_lib.ClusterSpec(cluster_spec), rpc_layer=rpc_layer),
      stream_output=stream_output,
      collective_leader=collective_leader)
  cluster.start()
  return cluster


class ParameterServerStrategyV2Test(test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(ParameterServerStrategyV2Test, cls).setUpClass()
    cluster_spec = {
        "worker": ["localhost:2223", "localhost:2224"],
        "ps": ["localhost:2222"]
    }
    cls.cluster = create_multi_process_cluster(cluster_spec)
    cls.cluster_resolver = cls.cluster.cluster_resolver
    # cls.strategy = DEParameterServerStrategy(cls.cluster_resolver)
    cls.strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
        cls.cluster_resolver)
    cls.coordinator = coordinator_lib.ClusterCoordinator(cls.strategy)

  @classmethod
  def tearDownClass(cls):
    super(ParameterServerStrategyV2Test, cls).tearDownClass()
    cls.cluster.stop()

  def testPerWorkerTraining(self):
    var_dtype = tf.dtypes.float32
    var_name = 'var'
    shape = [1]
    with self.strategy.scope():
      var = variables.Variable(initial_value=[0.0],
                               shape=shape,
                               dtype=var_dtype,
                               name=var_name,
                               per_worker_variable=True)
      var._trainable = True
    with backprop.GradientTape(persistent=True) as tape:

    # 定义训练步骤
    @tf.function
    def train_step():
      with tf.GradientTape() as tape:
        # var._maybe_create_per_worker_vars()
        value = var.read_value()
        # if not var.trainable:
        tape.watch(value) # still need this with var._trainable = True set.
        y = value * 2.0
      grad = tape.gradient(y, value)
      return grad

    @tf.function
    def train_step2():
      with tf.GradientTape() as tape:
        var._maybe_create_per_worker_vars()
        value = var.value()
        # if not var.trainable:
        tape.watch(value) # still need this with var._trainable = True set.
        y = value * 2.0
      grad = tape.gradient(y, value)
      return grad

    # 运行并检查结果
    grads = self.strategy.run(train_step2)
    print(f"grads :{grads}")
    print(f"var.read_all() {var.read_all()}")
  #@parameterized.parameters(True, False)
  # def testPerWorkerVariableCreation(self):
  #   var_dtype = tf.dtypes.float32
  #   var_name = 'var'
  #   shape = [1]  #if define_shape else None
  #
  #   with self.strategy.scope():
  #     var = variables.Variable(initial_value=[0.0],
  #                            shape=shape,
  #                            dtype=var_dtype,
  #                            name=var_name,
  #                            per_worker_de_variable=True)
  #
  #   # Use per-worker variable as a capture
  #   @def_function.function
  #   def worker_fn():
  #     var.assign_add(constant_op.constant([1.0]))
  #     return var
  #
  #   num_closures = 10
  #   for ix in range(num_closures):
  #     self.coordinator.schedule(worker_fn)
  #     # Read the PWV many times to ensure result is up-to-date
  #     self.coordinator.join()
  #     result_sum = sum(var.read_all()).numpy()
  #     self.assertEqual(result_sum, ix + 1)
  #
  #   for _ in range(num_closures):
  #     self.coordinator.schedule(worker_fn)
  #   self.coordinator.join()
  #
  #   # Verify placement of variables
  #   devices = [wv._get_values().device for wv in var._per_worker_vars._values]
  #   expected_devices = [
  #       f'/job:worker/replica:0/task:{ix}/device:CPU:0'
  #       for ix in range(self.strategy._num_workers)
  #   ]  # pylint: disable=protected-access
  #   self.assertAllEqual(devices, expected_devices)
  #
  #   result_sum = sum(var.read_all()).numpy()
  #   self.assertEqual(result_sum, num_closures * 2)

  # def testKerasFit(self):
  #   embed_dim = 8
  #   with self.strategy.scope():
  #     model = Sequential([
  #         layers.Input(shape=(1,), dtype=tf.int32),
  #         de.keras.layers.Embedding(embed_dim, key_dtype=tf.int32),
  #         layers.Flatten(),
  #         layers.Dense(1, activation='sigmoid')
  #     ])
  #     optimizer = Adam(1E-3)
  #     optimizer = de.DynamicEmbeddingOptimizer(optimizer)
  #     model.compile(loss='binary_crossentropy',
  #                   optimizer=optimizer,
  #                   metrics=['accuracy'])
  #
  #   ids = np.random.randint(0, 100, size=(64 * 2, 1))
  #   labels = np.random.randint(0, 2, size=(64 * 2, 1))
  #
  #   def dataset_fn(input_context):
  #     global_batch_size = 32
  #     batch_size = input_context.get_per_replica_batch_size(global_batch_size)
  #     dataset = tf.data.Dataset.from_tensor_slices((ids, labels))
  #     dataset = dataset.shard(input_context.num_input_pipelines,
  #                             input_context.input_pipeline_id)
  #     dataset = dataset.batch(batch_size).repeat()
  #     return dataset
  #
  #   dataset = self.strategy.distribute_datasets_from_function(dataset_fn)
  #
  #   history = model.fit(dataset, epochs=1, steps_per_epoch=len(ids) // 64)
  #   self.assertIn('loss', history.history)


# borrow from multi_process_lib._set_spawn_exe_path and modify it for tf_recommenders_addons
def custom_set_spawn_exe_path():
  if sys.argv[0].endswith('.py'):

    def guess_path(package_root):
      # If all we have is a python module path, we'll need to make a guess for
      # the actual executable path.
      if 'bazel-out' in sys.argv[0] and package_root in sys.argv[0]:
        package_root_base = sys.argv[0][:sys.argv[0].rfind(package_root)]
        binary = os.environ['TEST_TARGET'][2:].replace(':', '/', 1)
        print(f"package_root_base {package_root_base} binary {binary}")
        possible_path = os.path.join(package_root_base, package_root, binary)
        print('Guessed test binary path: %s', possible_path)
        if os.access(possible_path, os.X_OK):
          return possible_path
        return None

    path = guess_path('tf_recommenders_addons')
    if path is None:
      print('Cannot determine binary path. sys.argv[0]=%s os.environ=%s',
            sys.argv[0], os.environ)
      raise RuntimeError('Cannot determine binary path')
    sys.argv[0] = path
  # Note that this sets the executable for *all* contexts.
  multiprocessing.get_context().set_executable(sys.argv[0])


# This is not for pytest  bazel clean --expunge
# bazel test --test_output=all //tensorflow_recommenders_addons/dynamic_embedding/python/kernel_tests:parameter_server_bzl
if __name__ == "__main__":
  multi_process_lib._set_spawn_exe_path = custom_set_spawn_exe_path
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
