<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfra.dynamic_embedding


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>
<br/>
<br/>
<br/>
<br/>



Export dynamic_embedding APIs.



## Modules

[`math`](../tfra/dynamic_embedding/math.md) module: math operations.

## Classes

[`class CuckooHashTable`](../tfra/dynamic_embedding/CuckooHashTable.md): A generic mutable hash table implementation.

[`class FrequencyRestrictPolicy`](../tfra/dynamic_embedding/FrequencyRestrictPolicy.md): A derived policy to eliminate features in variable follow the

[`class GraphKeys`](../tfra/dynamic_embedding/GraphKeys.md): Extended standard names related to `dynamic_embedding_ops.Variable` to use

[`class ModelMode`](../tfra/dynamic_embedding/ModelMode.md): The global config of model modes.

[`class RestrictPolicy`](../tfra/dynamic_embedding/RestrictPolicy.md): Base class of restrict policies. Never use this class directly, but

[`class TimestampRestrictPolicy`](../tfra/dynamic_embedding/TimestampRestrictPolicy.md): A derived policy to eliminate features in variable follow the

[`class TrainableWrapper`](../tfra/dynamic_embedding/TrainableWrapper.md): This class is a trainable wrapper of Dynamic Embedding,

[`class Variable`](../tfra/dynamic_embedding/Variable.md): A Distributed version of HashTable(reference from lookup_ops.MutableHashTable)

## Functions

[`DynamicEmbeddingOptimizer(...)`](../tfra/dynamic_embedding/DynamicEmbeddingOptimizer.md): An optimizer wrapper to make any TensorFlow optimizer capable of training

[`embedding_lookup(...)`](../tfra/dynamic_embedding/embedding_lookup.md): Provides a dynamic version of embedding_lookup

[`embedding_lookup_sparse(...)`](../tfra/dynamic_embedding/embedding_lookup_sparse.md): Provides a dynamic version of embedding_lookup_sparse

[`embedding_lookup_unique(...)`](../tfra/dynamic_embedding/embedding_lookup_unique.md): Version of embedding_lookup that avoids duplicate lookups.

[`enable_inference_mode(...)`](../tfra/dynamic_embedding/enable_inference_mode.md): set inference mode.

[`enable_train_mode(...)`](../tfra/dynamic_embedding/enable_train_mode.md): enable train mode.

[`get_model_mode(...)`](../tfra/dynamic_embedding/get_model_mode.md): get model mode.

[`get_variable(...)`](../tfra/dynamic_embedding/get_variable.md): Gets an `Variable` object with this name if it exists,

[`safe_embedding_lookup_sparse(...)`](../tfra/dynamic_embedding/safe_embedding_lookup_sparse.md): Provides a dynamic version of `tf.nn.safe_embedding_lookup_sparse`.

