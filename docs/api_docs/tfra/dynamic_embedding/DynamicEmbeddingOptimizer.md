<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.DynamicEmbeddingOptimizer" />
<meta itemprop="path" content="Stable" />
</div>

# tfra.dynamic_embedding.DynamicEmbeddingOptimizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_optimizer.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>
<br/>
<br/>
<br/>
<br/>



An optimizer wrapper to make any TensorFlow optimizer capable of training

``` python
tfra.dynamic_embedding.DynamicEmbeddingOptimizer()
```



<!-- Placeholder for "Used in" -->
Dynamic Embeddding Variables.

#### Args:


* <b>`self`</b>: a TensorFlow optimizer.


#### Example usage:


```python
optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(
    tf.train.AdamOptimizer(0.001))
```



#### Returns:

The optimizer itself but has ability to train Dynamic Embedding Variables.
