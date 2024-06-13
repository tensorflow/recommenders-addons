<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.HkvEvictStrategy" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tfra.dynamic_embedding.HkvEvictStrategy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/recommenders-addons/blob/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_creator.py#L141">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>
<br/>
<br/>
<br/>
<br/>



## Class `HkvEvictStrategy`





<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/blob/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_creator.py#L141">View source</a>

``` python
@unique
class HkvEvictStrategy(IntEnum):
  LRU = 0
  LFU = 1
  EPOCHLRU = 2
  EPOCHLFU = 3
  CUSTOMIZED = 4
```

The HkvEvictStrategy contains 5 enumeration options that represent different hkv evict strategies. Use LRU default.

HKV uses `scores` to implement the evict strategy.The key-value with smaller scores will be evicted first. Therefore, all supported evict strategies are implemented by defining different scores. For example, For example, the scores Definition of LRU is：`Device clock in a nanosecond, which could differ slightly from host clock.`This ensures that your newly inserted or updated key-value has the largest socres, and the oldest key-value has the smaller socres. This is consistent with the LRU algorithm.More information can be found [hkv-evict-strategy](https://github.com/LinGeLin/HierarchicalKV?tab=readme-ov-file#evict-strategy).


#### 	Definition of Score:


* <b>`LRU`</b>: Device clock in a nanosecond, which could differ slightly from host clock. Socres get bigger and bigger over time。
* <b>`LFU`</b>: Frequency increment provided by caller via the input parameter of scores of insert-like APIs as the increment of frequency. Frequency increment here is set to 1. So the first time key-value is inserted, scores will be 1, and scores will be increased by 1 with each subsequent update
* <b>`EPOCHLRU`</b>: The high 32bits is the global epoch provided via the input parameter of global_epoch,
the low 32bits is equal to (device_clock >> 20) & 0xffffffff with granularity close to 1 ms.
* <b>`EPOCHLFU`</b>: The high 32bits is the global epoch provided via the input parameter of global_epoch,
the low 32bits is the frequency,
the frequency will keep constant after reaching the max value of 0xffffffff.

When you choose EPOCHLRU or EPOCHLFU, you need to set `step_per_epoch` in `HkvHashTableConfig` to indicate the number of steps in each epoch. `global_epoch` will be automatically initialized to 0. Add 1 automatically when you update the `step_per_epoch` step

* <b>`CUSTOMIZED`</b>: Fully provided by the caller via the input parameter of scores of insert-like APIs.

If you want to customize scores, you can choose `CUSTOMIZED` strategy. You also need to set `gen_scores_fn` in `HkvHashTableConfig` to be your own implementation of the scores generation method. The input to this function is keys and the output is scores. Such as:
```
def gen_scores_fn(keys):
  return tf.add(keys, tf.constant([1], shape=[4,], dtype=dtypes.int64))
```


