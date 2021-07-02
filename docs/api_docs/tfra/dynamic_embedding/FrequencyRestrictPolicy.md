<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.FrequencyRestrictPolicy" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="status"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="apply_restriction"/>
<meta itemprop="property" content="apply_update"/>
</div>

# tfra.dynamic_embedding.FrequencyRestrictPolicy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/restrict_policies.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>
<br/>
<br/>
<br/>
<br/>



## Class `FrequencyRestrictPolicy`

A derived policy to eliminate features in variable follow the

Inherits From: [`RestrictPolicy`](../../tfra/dynamic_embedding/RestrictPolicy.md)

<!-- Placeholder for "Used in" -->
`lowest-occurrence-out-first` rule.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/restrict_policies.py">View source</a>

``` python
__init__(var)
```

A frequency status sparse variable is created. The frequency status has
same key_dtype as the target variable and value_dtype in `int32`, which
indicates the occurrence times of the feature.

#### Args:


* <b>`var`</b>: A <a href="../../tfra/dynamic_embedding/Variable.md"><code>dynamic_embedding.Variable</code></a> object to be restricted.



## Properties

<h3 id="status"><code>status</code></h3>

Get status variable which save information about properties of features.




## Methods

<h3 id="apply_restriction"><code>apply_restriction</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/restrict_policies.py">View source</a>

``` python
apply_restriction(
    num_reserved,
    **kwargs
)
```

Define the rule to restrict the size of the target variable by eliminating
k features with least occurrence, and number of `num_reserved` features will
be left.

#### Args:


* <b>`num_reserved`</b>: int. Number of remained keys after restriction.
* <b>`**kwargs`</b>: (Optional) reserved keyword arguments.
  trigger: int. The triggered threshold to execute restriction. Default
    equals to `num_reserved`.


#### Returns:

An operation to do restriction.


<h3 id="apply_update"><code>apply_update</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/restrict_policies.py">View source</a>

``` python
apply_update(ids)
```

Define the rule to update the frequency status. If any feature shows up,
then its frequency value will be increased by 1.

#### Args:


* <b>`ids`</b>: A Tensor. Keys appear in training. These keys in status variable
  will be updated if needed.


#### Returns:

An operation to update timestamp status.




