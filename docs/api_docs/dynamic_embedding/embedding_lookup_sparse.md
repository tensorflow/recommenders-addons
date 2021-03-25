<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.embedding_lookup_sparse" />
<meta itemprop="path" content="Stable" />
</div>

# tfra.dynamic_embedding.embedding_lookup_sparse

``` python
tfra.dynamic_embedding.embedding_lookup_sparse(
    params,
    sp_ids,
    sp_weights,
    partition_strategy=None,
    name='embedding_lookup_sparse',
    combiner='mean',
    max_norm=None,
    return_trainable=False
)
```

Provides a dynamic version of embedding_lookup_sparse
  similar with tf.nn.embedding_lookup_sparse.

This op assumes that there is at least one id for each row in the dense tensor
represented by sp_ids (i.e. there are no rows with empty features), and that
all the indices of sp_ids are in canonical row-major order.

It also assumes that all id values lie in the range [0, p0), where p0
is the sum of the size of params along dimension 0.

#### Args:

* <b>`params`</b>: A single `dynamic_embedding.Variable` instance representing
    the complete embedding tensor.
* <b>`sp_ids`</b>: N x M `SparseTensor` of int64 ids where N is typically batch size
    and M is arbitrary.
* <b>`sp_weights`</b>: either a `SparseTensor` of float / double weights, or `None` to
    indicate all weights should be taken to be 1. If specified, `sp_weights`
    must have exactly the same shape and indices as `sp_ids`.
* <b>`partition_strategy`</b>: No used.
* <b>`name`</b>: Optional name for the op.
* <b>`combiner`</b>: A string specifying the reduction op. Currently "mean", "sqrtn"
    and "sum" are supported. "sum" computes the weighted sum of the embedding
    results for each row. "mean" is the weighted sum divided by the total
    weight. "sqrtn" is the weighted sum divided by the square root of the sum
    of the squares of the weights.
* <b>`max_norm`</b>: If not `None`, each embedding is clipped if its l2-norm is larger
    than this value, before combining.
* <b>`return_trainable`</b>: optional, If True, also return TrainableWrapper create by
    `dynamic_embedding.embedding_lookup`


#### Returns:

* <b>`combined_embeddings`</b>: A dense tensor representing the combined embeddings
    for the sparse ids. For each row in the dense tensor represented by
    `sp_ids`, the op looks up the embeddings for all ids in that row,
    multiplies them by the corresponding weight, and combines these embeddings
    as specified.

    In other words, if

      `shape(combined params) = [+infinity, dim]`

    and

      `shape(sp_ids) = shape(sp_weights) = [d0, d1, ..., dn]`

    then

      `shape(output) = [d0, dim]`.

    For instance, if params dim=20, and sp_ids / sp_weights are

      ```python
      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id 0, weight 1.0
      [2, 3]: id 1, weight 3.0
      ```

    with `combiner`="mean", then the output will be a 3x20 matrix where

      ```python
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = (params[0, :] * 1.0) / 1.0
      output[2, :] = (params[1, :] * 3.0) / 3.0
      ```
* <b>`trainable_wrap`</b>:     A TrainableWrapper object used to fill the Optimizers `var_list`
      Only provided if `return_trainable` is True.

#### Raises:

* <b>`TypeError`</b>: If `sp_ids` is not a `SparseTensor`, or if `sp_weights` is
    neither `None` nor `SparseTensor`.
* <b>`ValueError`</b>: If `combiner` is not one of {"mean", "sqrtn", "sum"}.