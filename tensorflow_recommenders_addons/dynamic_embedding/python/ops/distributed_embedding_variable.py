from tensorflow_recommenders_addons.dynamic_embedding.python.ops.embedding_weights import EmbeddingWeights
from tensorflow.python.distribute import values as distribute_values_lib

import tensorflow as tf


class DistributedVariableWrapper(EmbeddingWeights,
                                 distribute_values_lib.DistributedVariable):

  def __init__(self, strategy, values, aggregation, var_policy=None):
    super(DistributedVariableWrapper, self).__init__(strategy, values,
                                                     aggregation, var_policy)
    self.shadow = self._get_on_device_or_primary()

  def verify_embedding_weights(self, sparse_ids, sparse_weights=None):
    EmbeddingWeights.verify_embedding_param_weights(self.shadow.params,
                                                    sparse_ids, sparse_weights)

  def embedding_lookup(self,
                       ids,
                       name=None,
                       max_norm=None) -> (tf.Tensor, EmbeddingWeights):
    raise NotImplementedError("embedding_lookup is not supported in "
                              "DistributedVariableWrapper")