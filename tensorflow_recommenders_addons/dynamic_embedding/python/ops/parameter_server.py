from tensorflow.python.distribute import ps_values, distribute_lib
from tensorflow.python.distribute.distribute_lib import _get_per_thread_mode
from tensorflow.python.distribute.parameter_server_strategy_v2 import ParameterServerStrategyV2, \
  ParameterServerStrategyV2Extended
from tensorflow.python.ops import variables
import tensorflow as tf



class DEPerWorkerVariable(ps_values.PerWorkerVariable):
  def __init__(self, *args, **kwargs):
    super(DEPerWorkerVariable, self).__init__(*args, **kwargs)

def create_per_worker_de_variable(strategy, name, dtype, shape):
  # printop = tf.print("st_2:", strategy,
  #                    tf.distribute.get_replica_context() ,
  #                    output_stream=tf.compat.v1.logging.error)
  # with tf.control_dependencies([printop]):
    with strategy.scope():
      return variables.Variable(initial_value=(),
        shape=shape, dtype=dtype, name=name,
        per_worker_de_variable=True)

def create_ps_trainable_wrapper(strategy, params, ids, max_norm, initial_value, **kwargs):
  with strategy.scope():
    return variables.Variable(initial_value=initial_value,
                              params=params, max_norm=max_norm, ids=ids,
                              ps_trainable_wrapper=True, **kwargs)

def create_ps_shadow_variable(strategy, params, trainable, max_norm, **kwargs):
  with strategy.scope():
    return variables.Variable(distribute_strategy=strategy,
                              params=params, max_norm=max_norm, trainable=trainable,
                              ps_shadow_variable=True, **kwargs)

original_create_variable = ParameterServerStrategyV2Extended._create_variable

def patched_create_variable(self, next_creator, **kwargs):
  if kwargs.pop("per_worker_de_variable", False):
    return _create_per_worker_de_variable(self, next_creator, **kwargs)
  if kwargs.pop("ps_trainable_wrapper", False):
    return _create_ps_trainable_wrapper(self, next_creator, **kwargs)
  if kwargs.pop("ps_shadow_variable", False):
    return _create_ps_shadow_variable(self, next_creator, **kwargs)
  return original_create_variable(self, next_creator, **kwargs)

def _create_ps_trainable_wrapper(strategy_extended, next_creator, **kwargs):
  from tensorflow_recommenders_addons.dynamic_embedding import TrainableWrapper
  return TrainableWrapper(strategy_extended._container_strategy(), next_creator, **kwargs)

def _create_ps_shadow_variable(strategy_extended, next_creator, **kwargs):
  from tensorflow_recommenders_addons.dynamic_embedding.python.ops.shadow_embedding_ops import ShadowVariable
  return ShadowVariable(next_creator, **kwargs)

def _create_per_worker_de_variable(strategy_extended, next_creator, **kwargs):
  return DEPerWorkerVariable(strategy_extended._container_strategy(), next_creator, **kwargs)

ParameterServerStrategyV2Extended._create_variable = patched_create_variable

class DEParameterServerStrategy(ParameterServerStrategyV2):
  def __init__(self, cluster_resolver, variable_partitioner=None):
    super(DEParameterServerStrategy, self).__init__(cluster_resolver, variable_partitioner)