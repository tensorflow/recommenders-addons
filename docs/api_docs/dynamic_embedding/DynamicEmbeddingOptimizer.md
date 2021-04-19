<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.DynamicEmbeddingOptimizer" />
<meta itemprop="path" content="Stable" />
</div>

# tfra.dynamic_embedding.DynamicEmbeddingOptimizer

``` python
tfra.dynamic_embedding.DynamicEmbeddingOptimizer()
```

An optimizer wrapper to make any TensorFlow optimizer capable of training
Dynamic Embeddding Variables.

#### Args:

* <b>`self`</b>: a TensorFlow optimizer.

Example usage:

  ```python
  optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(
      tf.train.AdamOptimizer(0.001))
  ```


#### Returns:

The optimizer itself but has ability to train Dynamic Embedding Variables.