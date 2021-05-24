<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.math.sparse_segment_sum" />
<meta itemprop="path" content="Stable" />
</div>

# tfra.dynamic_embedding.math.sparse_segment_sum

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



Computes the sum along sparse segments of a tensor. It do same things

``` python
tfra.dynamic_embedding.math.sparse_segment_sum(
    data,
    indices,
    segment_ids,
    name=None,
    num_segments=None
)
```



<!-- Placeholder for "Used in" -->
as `tf.sparse.segment_sum`. Here we provide GPU impl.

Go [tf api](https://www.tensorflow.org/api_docs/python/tf/sparse/segment_sum)
for more details.

#### Args:


* <b>`data`</b>: A `Tensor` with data that will be assembled in the output.
* <b>`indices`</b>: A 1-D `Tensor` with indices into `data`. Has same rank as
  `segment_ids`.
* <b>`segment_ids`</b>: A 1-D `Tensor` with indices into the output `Tensor`. Values
  should be sorted and can be repeated.
* <b>`name`</b>: A name for the operation (optional).
* <b>`num_segments`</b>: An optional int32 scalar. Indicates the size of the output
  `Tensor`.


#### Returns:

A `tensor` of the shape as data, except for dimension 0 which
has size `k`, the number of segments specified via `num_segments` or
inferred for the last element in `segments_ids`.
