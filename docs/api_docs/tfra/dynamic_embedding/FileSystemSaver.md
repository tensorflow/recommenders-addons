<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.FileSystemSaver" />
<meta itemprop="property" content="__init__"/>
</div>

# tfra.dynamic_embedding.FileSystemSaver

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_creator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>
<br/>
<br/>
<br/>
<br/>



## Class `FileSystemSaver`

A saver for independently saving the keys and values of DynamicEmbedding into filesystem by patching TensorFlow saving function.
This saver is also supported expanding and shrinking the number of tables in distributed training.
User would not need to change any code style, just simply use TensorFlow savedmodel or checkpoint API as normal.

Inherits From: `DynamicEmbeddingSaver`

<!-- Placeholder for "Used in" -->

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_creator.py">View source</a>

``` python
__init__(config)
```

A saver for dynamic variable is created. 
Which could be as a parameter saver for kv_creator function.

If there is no config `save_path` parameter for `FileSystemSaver`, default saving path is `"{model_path}/variables/TFRADynamicEmbedding"`.

#### Args:


* <b>`proc_size`</b>: A int parameter to config the size of global running TensorFlow instance(process), which usually set during MPI training.
* <b>`proc_rank`</b>: A int parameter to config the rank of local runtime TensorFlow instance(process), which usually set during MPI training.
* <b>`save_path`</b>: A string parameter for specific KV files saving path.
* <b>`buffer_size`</b>: A int parameter to set writing/reading how many keys at a time when handle KV files.



## Useage
``` python
import tensorflow_recommenders_addons.dynamic_embedding as de

test_devices = ['GPU:0', 'GPU:1']
params0 = de.get_variable(
            'table0',
            devices=test_devices,
            dim=8,
            initializer=0.0,
            bp_v2=True,
            kv_creator=de.CuckooHashTableCreator(
              saver=de.FileSystemSaver()
            )

params1 = de.get_variable(
            'table1',
            devices=test_devices,
            dim=8,
            initializer=0.0,
            bp_v2=True,
            kv_creator=de.CuckooHashTableCreator(
              saver=de.FileSystemSaver(
                save_path='/tmp/TFRADynamicEmbedding'
              )
            )
```





