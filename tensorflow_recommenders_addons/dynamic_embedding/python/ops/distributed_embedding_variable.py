from tensorflow_recommenders_addons.dynamic_embedding.python.ops.embedding_variable import IEmbeddingVariable
from tensorflow.python.distribute import values as distribute_values_lib


class DistributedVariableWrapper(IEmbeddingVariable,
                                 distribute_values_lib.DistributedVariable):

  def __init__(self, strategy, values, aggregation, var_policy=None):
    super(DistributedVariableWrapper, self).__init__(strategy, values,
                                                     aggregation, var_policy)
    self.shadow = values[0]

  def verify_embedding_weights(self, sparse_ids, sparse_weights=None):
    IEmbeddingVariable.verify_embedding_param_weights(self.shadow.params,
                                                      sparse_ids,
                                                      sparse_weights)
