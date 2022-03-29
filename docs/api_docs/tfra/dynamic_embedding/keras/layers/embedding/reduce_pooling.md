<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.keras.layers.embedding.reduce_pooling" />
<meta itemprop="path" content="Stable" />
</div>

# tfra.dynamic_embedding.keras.layers.embedding.reduce_pooling

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/keras/layers/embedding.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>
<br/>
<br/>
<br/>
<br/>



Default combine_fn for Embedding layer. By assuming input

``` python
@tf.function
tfra.dynamic_embedding.keras.layers.embedding.reduce_pooling(
    x,
    combiner='sum'
)
```



<!-- Placeholder for "Used in" -->
ids shape is (batch_size, s1, ..., sn), it will get lookup result
with shape (batch_size, s1, ..., sn, embedding_size). Every
sample in a batch will be reduecd to a single vector, and thus
the output shape is (batch_size, embedding_size)