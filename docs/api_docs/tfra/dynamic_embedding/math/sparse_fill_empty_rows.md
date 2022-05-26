<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.math.sparse_fill_empty_rows" />
<meta itemprop="path" content="Stable" />
</div>

# tfra.dynamic_embedding.math.sparse_fill_empty_rows

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



Fills empty rows in the input 2-D `SparseTensor` with a default value. 

``` python
tfra.dynamic_embedding.math.sparse_fill_empty_rows(
    sp_input,
    default_value,
    name=None
)
```



<!-- Placeholder for "Used in" -->

It does same things as `tf.sparse.fill_empty_rows`. Here we provide GPU implement.

Go [TF API](https://www.tensorflow.org/api_docs/python/tf/sparse/fill_empty_rows)
for more details.

#### Args:


* <b>`sp_input`</b>: A `SparseTensor` with shape `[N, M]`.
* <b>`default_value`</b>: The value to fill for empty rows, with the same type as
  `sp_input`.
* <b>`name`</b>: A name prefix for the returned tensors (optional).


#### Returns:


* <b>`sp_ordered_output`</b>: A `SparseTensor` with shape `[N, M]`, and with all empty
  rows filled in with `default_value`.
* <b>`empty_row_indicator`</b>: A bool vector of length `N` indicating whether each
  input row was empty.