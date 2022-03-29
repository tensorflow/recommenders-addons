<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.shadow_ops.embedding_lookup" />
<meta itemprop="path" content="Stable" />
</div>

# tfra.dynamic_embedding.shadow_ops.embedding_lookup

<!-- Insert buttons and diff -->

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



Shadow version of dynamic_embedding.embedding_lookup. It use existed shadow

``` python
tfra.dynamic_embedding.shadow_ops.embedding_lookup(
    shadow,
    ids,
    partition_strategy=None,
    name=None,
    validate_indices=None
)
```



<!-- Placeholder for "Used in" -->
variable to to embedding lookup, and store the result. No by-product will
be introduced in this call. So it can be decorated by `tf.function`.

#### Args:


* <b>`shadow`</b>: A ShadowVariable object.
* <b>`ids`</b>: A tensor with any shape as same dtype of params.key_dtype.
* <b>`partition_strategy`</b>: No used, for API compatiblity with `nn.emedding_lookup`.
* <b>`name`</b>: A name for the operation.
* <b>`validate_indices`</b>: No used, just for compatible with nn.embedding_lookup .


#### Returns:

A tensor with shape [shape of ids] + [dim],
  dim is equal to the value dim of params.
  containing the values from the params tensor(s) for keys in ids.
