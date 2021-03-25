<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.safe_embedding_lookup_sparse" />
<meta itemprop="path" content="Stable" />
</div>

# tfra.dynamic_embedding.safe_embedding_lookup_sparse

``` python
tfra.dynamic_embedding.safe_embedding_lookup_sparse(
    embedding_weights,
    sparse_ids,
    sparse_weights=None,
    combiner='mean',
    default_id=None,
    name='safe_embedding_lookup_sparse',
    partition_strategy=None,
    max_norm=None,
    return_trainable=False
)
```

Provides a dynamic version of <a href="../../tf/nn/safe_embedding_lookup_sparse.md"><code>tf.nn.safe_embedding_lookup_sparse</code></a>.

Lookup embedding results, accounting for empty features and invalid weights.

Any IDs will be treated as valid include non-positive IDs.
Invalid weights (<= 0) are pruned from input weights, as well as any IDs
with non-positive weight. For an entry with no features, the embedding vector
for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

The ids and weights may be multi-dimensional. Embeddings are always aggregated
along the last dimension.

#### Args:

* <b>`embedding_weights`</b>: A single `dynamic_embedding.Variable` instance
    representing the complete embedding tensor.
* <b>`sparse_ids`</b>: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
    ids. `d_0` is typically batch size.
* <b>`sparse_weights`</b>: `SparseTensor` of same shape as `sparse_ids`, containing
    float weights corresponding to `sparse_ids`, or `None` if all weights are
    be assumed to be 1.0.
* <b>`combiner`</b>: A string specifying how to combine embedding results for each
    entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean" the
    default.
* <b>`default_id`</b>: The id to use for an entry with no features.
* <b>`name`</b>: A name for this operation (optional).
* <b>`partition_strategy`</b>: A string specifying the partitioning strategy. Currently
    `"div"` and `"mod"` are supported. Default is `"div"`.
* <b>`max_norm`</b>: If not `None`, all embeddings are l2-normalized to max_norm before
    combining.


#### Returns:

* <b>`combined_embeddings`</b>:     A dense `Tensor` of shape `[d_0, d_1, ..., d_{n-1}, e_1, ..., e_m]`.
* <b>`trainable_wrap`</b>:     A TrainableWrapper object used to fill the Optimizers `var_list`
      Only provided if `return_trainable` is True.


#### Raises:

* <b>`ValueError`</b>: if `embedding_weights` is empty.