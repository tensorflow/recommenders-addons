<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.RestrictPolicy" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="status"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="apply_restriction"/>
<meta itemprop="property" content="apply_update"/>
</div>

# tfra.dynamic_embedding.RestrictPolicy

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



## Class `RestrictPolicy`

Base class of restrict policies. Never use this class directly, but



<!-- Placeholder for "Used in" -->
instead of of its derived class.

This class defines the rules for tracking and restricting the size of the
<a href="../../tfra/dynamic_embedding/Variable.md"><code>dynamic_embedding.Variable</code></a>. If the variable joins training via stateful
optimizer, the policy also manage the slots of the optimizer. It could own
a status to keep tracking the state of affairs of the features presented
in sparse <a href="../../tfra/dynamic_embedding/Variable.md"><code>dynamic_embedding.Variable</code></a>.

`RestrictPolicy` requires a set of methods to be override, in its derived
policies: `apply_update`, `apply_restriction`, `status`.

* apply_update: keep tracking on the status of the sparse variable,
    specifically the attributes of each key in sparse variable.
* apply_restriction: eliminate the features which are not legitimate.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/restrict_policies.py">View source</a>

``` python
__init__(var)
```

Create a new RestrictPolicy.


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

Define the rule to restrict the size of the target variable. There are
three kinds of variables are token into consideration: variable, status
variable, and variables in slots. Number of `num_reserved` features will
be kept in variable.

#### Args:


* <b>`num_reserved`</b>: number of remained keys after restriction.
* <b>`**kwargs`</b>: (Optional) reserved keyword arguments.


#### Returns:

An operation to restrict the sizes of variable and variables in slots.


<h3 id="apply_update"><code>apply_update</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/restrict_policies.py">View source</a>

``` python
apply_update(ids)
```

Define the rule to update status, with tracking the
changes in variable and slots.

#### Args:


* <b>`ids`</b>: A Tensor. Keys appear in training. These keys in status
  variable will be updated if needed.


#### Returns:

An operation to update status.




