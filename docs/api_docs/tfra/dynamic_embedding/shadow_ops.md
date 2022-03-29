<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.shadow_ops" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfra.dynamic_embedding.shadow_ops


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/shadow_embedding_ops.py">
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

`The file will be introduced as `shadow_ops` under `dynamic_embedding`.
It is a submodule of `dynamic_embedding`.

In TensorFlow 2.x, tf.function is introduced to speedup the computation.
And also modular programming based on [tf.Module](https://www.tensorflow.org/guide/intro_to_modules)
are recommended because of the Pythonic style APIs. But APIs like
`embedding_lookup`, `embedding_lookup_unique`, `embedding_lookup_sparse`, and
`safe_embedding_lookup_sparse` in `dynamic_embedding`, are wrappers of
`embedding_lookup`. And it will create a TrainableWrapper object inside
the function, which doesn't meet the requirements of
[tf.function](https://www.tensorflow.org/guide/function)

The `shadow_ops` submodule is designed to support usage on `tf.function`
and modular style development, like keras.

## Classes

[`class ShadowVariable`](../../tfra/dynamic_embedding/shadow_ops/ShadowVariable.md): ShadowVariable is a eager persistent twin of TrainableWrapper.

## Functions

[`embedding_lookup(...)`](../../tfra/dynamic_embedding/shadow_ops/embedding_lookup.md): Shadow version of dynamic_embedding.embedding_lookup. It use existed shadow

