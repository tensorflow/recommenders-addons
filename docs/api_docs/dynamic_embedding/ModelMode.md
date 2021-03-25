<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.ModelMode" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="CURRENT_SETTING"/>
<meta itemprop="property" content="INFERENCE"/>
<meta itemprop="property" content="TRAIN"/>
</div>

# tfra.dynamic_embedding.ModelMode

## Class `ModelMode`



The global config of model modes.

  For `TrainableWrapper` is not multi-threads safe that causes threads
  competition and Out-Of-Bound exception in highly concurrent
  inference scenario. So we define the `ModelMode` APIs to instruct
  the `TrainableWrapper` to build different multi-threads safe sub-graph
  for 'TrainableWrapper.read_values' on inference mode.

The following standard modes are defined:

* `TRAIN`: training/fitting mode.
* `INFERENCE`: predication/inference mode.

## Class Members

<h3 id="CURRENT_SETTING"><code>CURRENT_SETTING</code></h3>

<h3 id="INFERENCE"><code>INFERENCE</code></h3>

<h3 id="TRAIN"><code>TRAIN</code></h3>

