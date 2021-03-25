<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.embedding_lookup" />
<meta itemprop="path" content="Stable" />
</div>

# tfra.dynamic_embedding.embedding_lookup

``` python
tfra.dynamic_embedding.embedding_lookup(
    params,
    ids,
    partition_strategy=None,
    name=None,
    validate_indices=None,
    max_norm=None,
    return_trainable=False
)
```

Provides a dynamic version of embedding_lookup
  similar with tf.nn.embedding_lookup.

Ids are flattened to a 1d tensor before being passed to embedding_lookup
then, they are unflattend to match the original ids shape plus an extra
leading dimension of the size of the embeddings.

#### Args:

* <b>`params`</b>: A dynamic_embedding.Variable instance.
* <b>`ids`</b>: a tensor with any shape as same dtype of params.key_dtype.
* <b>`partition_strategy`</b>: No used, for API compatiblity with `nn.emedding_lookup`.
* <b>`name`</b>: A name for the operation (optional).
* <b>`validate_indices`</b>: No used, just for compatible with nn.embedding_lookup .
* <b>`max_norm`</b>: If not `None`, each embedding is clipped if its l2-norm is larger
    than this value.
* <b>`return_trainable`</b>: optional, If True, also return TrainableWrapper

#### Returns:

A tensor with shape [shape of ids] + [dim],
  dim is equal to the value dim of params.
  containing the values from the params tensor(s) for keys in ids.
* <b>`trainable_wrap`</b>:     A TrainableWrapper object used to fill the Optimizers `var_list`
      Only provided if `return_trainable` is True.