<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.math.sparse_reshape" />
<meta itemprop="path" content="Stable" />
</div>

# tfra.dynamic_embedding.math.sparse_reshape

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/math_ops.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>
<br/>
<br/>
<br/>
<br/>



Reshapes a `SparseTensor` to represent values in a new dense shape.

``` python
tfra.dynamic_embedding.math.sparse_reshape(
    sp_input,
    shape,
    name=None
)
```



<!-- Placeholder for "Used in" -->

It does same things as `tf.sparse.reshape`. Here we provide GPU implement.

Go [TF API](https://www.tensorflow.org/api_docs/python/tf/sparse/reshape)
for more details.

#### Args:


* <b>`sp_input`</b>: The input `SparseTensor`.
* <b>`shape`</b>: A 1-D (vector) int64 `Tensor` specifying the new dense shape of the
  represented `SparseTensor`.
* <b>`name`</b>: A name prefix for the returned tensors (optional).


#### Returns:

A `SparseTensor` with the same non-empty values but with indices calculated
by the new dense shape.
