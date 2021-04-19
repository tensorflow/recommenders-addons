<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__all__"/>
</div>

# Module: tfra.dynamic_embedding

Public API for tfra.dynamic_embedding namespace.

## Classes

[`class Variable`](./dynamic_embedding/Variable.md): A Distributed version of HashTable(reference from lookup_ops.MutableHashTable)

[`class ModelMode`](./dynamic_embedding/ModelMode.md): The global config of model modes.

[`class TrainableWrapper`](./dynamic_embedding/TrainableWrapper.md): This class is a trainable wrapper of Dynamic Embedding,

## Functions

[`get_variable(...)`](./dynamic_embedding/get_variable.md): Gets an [`Variable`](./dynamic_embedding/Variable.md) object with this name if it exists,

[`embedding_lookup(...)`](./dynamic_embedding/embedding_lookup.md): Provides a dynamic version of <a href="https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup"><code>tf.nn.embedding_lookup</code></a>

[`embedding_lookup_sparse(...)`](./dynamic_embedding/embedding_lookup_sparse.md): Provides a dynamic version of <a href="https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup_sparse"><code>tf.nn.embedding_lookup_sparse</code></a>

[`safe_embedding_lookup_sparse(...)`](./dynamic_embedding/safe_embedding_lookup_sparse.md): Provides a dynamic version of <a href="https://www.tensorflow.org/api_docs/python/tf/nn/safe_embedding_lookup_sparse"><code>tf.nn.safe_embedding_lookup_sparse</code></a>.

[`DynamicEmbeddingOptimizer(...)`](./dynamic_embedding/DynamicEmbeddingOptimizer.md): An optimizer wrapper to make any TensorFlow optimizer capable of training [`Variable`](./dynamic_embedding/Variable.md).

[`enable_train_mode(...)`](./dynamic_embedding/enable_train_mode.md): enable train mode.

[`enable_inference_mode(...)`](./dynamic_embedding/enable_inference_mode.md): set inference mode.

[`get_model_mode(...)`](./dynamic_embedding/get_model_mode.md): get model mode.
