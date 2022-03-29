<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.keras.layers.embedding" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfra.dynamic_embedding.keras.layers.embedding


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



Dynamic Embedding is designed for Large-scale Sparse Weights Training.

See [Sparse Domain Isolation](https://github.com/tensorflow/community/pull/237)

## Classes

[`class BasicEmbedding`](../../../../tfra/dynamic_embedding/keras/layers/BasicEmbedding.md): A keras style Embedding layer. The `BasicEmbedding` layer acts same like

[`class FieldWiseEmbedding`](../../../../tfra/dynamic_embedding/keras/layers/FieldWiseEmbedding.md): An embedding layer, which feature ids are mapped into fields.

[`class SquashedEmbedding`](../../../../tfra/dynamic_embedding/keras/layers/SquashedEmbedding.md): The SquashedEmbedding layer allow arbirary input shape of feature ids, and get

## Functions

[`reduce_pooling(...)`](../../../../tfra/dynamic_embedding/keras/layers/embedding/reduce_pooling.md): Default combine_fn for Embedding layer. By assuming input

