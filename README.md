# TensorFlow Recommenders Addons
-----------------
![TensorFlow Recommenders logo](assets/SIGRecommendersAddons.png)
[![PyPI Status Badge](https://badge.fury.io/py/tensorflow-recommenders-addons.svg)](https://pypi.org/project/tensorflow-recommenders-addons/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tensorflow-recommenders-addons)](https://pypi.org/project/tensorflow-recommenders-addons/)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](docs/api_docs/)


TensorFlow Recommenders Addons are a collection of projects related to
large-scale recommendation systems built upon TensorFlow. They are contributed
and maintained by the community. Those contributions will be complementary to
TensorFlow Core and TensorFlow Recommenders etc.

## Scope

See approved TensorFlow RFC #[313](https://github.com/tensorflow/community/pull/313). 

TensorFlow has open-sourced [TensorFlow Recommenders](https://blog.tensorflow.org/2020/09/introducing-tensorflow-recommenders.html)
([github.com/tensorflow/recommenders](http://github.com/tensorflow/recommenders)),
an open-source TensorFlow package that makes building, evaluating, and serving
sophisticated recommender models easy.

Further, this repo is maintained by TF SIG Recommenders
([recommenders@tensorflow.org](https://groups.google.com/a/tensorflow.org/g/recommenders))
for community contributions. SIG Recommenders can contributes more addons as complementary
to TensorFlow Recommenders, or any helpful libraries related to recommendation systems using
TensorFlow. The contribution areas can be broad and don't limit to the topic listed below:

* Training with scale: How to train from super large sparse features? How to
deal with dynamic embedding?
* Serving with efficiency: Given recommendation models are usually pretty
large, how to serve super large models easily, and how to serve efficiently?
* Modeling with SoTA techniques: online learning, multi-target learning, deal
with quality inconsistent among online and offline, model understandability,
GNN etc.
* End-to-end pipeline: how to train continuously, e.g. integrate with platforms
like TFX.
* Vendor specific extensions and platform integrations: for example, runtime
specific frameworks (e.g. NVIDIA Merlin, …), and integrations with Cloud services
(e.g. GCP, AWS, Azure…)

## RFCs
* [RFC: Dynamic Embedding](rfcs/20200424-sparse-domain-isolation.md)
* [RFC: Embedding Variable](https://docs.google.com/document/d/1odez6-69YH-eFcp8rKndDHTNGxZgdFFRJufsW94_gl4/edit)

## Recommenders-Addons Subpackages

* [tfra.dynamic_embedding](docs/api_docs/dynamic_embedding.md)
* [tfra.embedding_variable](https://github.com/tensorflow/recommenders-addons/blob/master/docs/tutorials/embedding_variable_tutorial.ipynb)

## Tutorials
See [`docs/tutorials/`](docs/tutorials/) for end-to-end examples of each subpackages.

## Maintainership

We adopt proxy maintainership as in [TensorFlow Recommenders-Addons](https://github.com/tensorflow/recommenders-addons):

*Projects and subpackages are compartmentalized and each is maintained by those
with expertise and vested interest in that component.*

*Subpackage maintainership will only be granted after substantial contribution
has been made in order to limit the number of users with write permission.
Contributions can come in the form of issue closings, bug fixes, documentation,
new code, or optimizing existing code. Submodule maintainership can be granted
with a lower barrier for entry as this will not include write permissions to
the repo.*

## Installation
#### Stable Builds
TensorFlow Recommenders-Addons is available on PyPI for Linux, macOS. To install the latest version, 
run the following:
```
pip install tensorflow-recommenders-addons
```

To ensure you have a version of TensorFlow that is compatible with TensorFlow Recommenders-Addons, 
you can specify the `tensorflow` extra requirement during install:

```
pip install tensorflow-recommenders-addons[tensorflow]
```

Similar extras exist for the `tensorflow-gpu` and `tensorflow-cpu` packages.
 

To use TensorFlow Recommenders-Addons:

```python
import tensorflow as tf
import tensorflow_recommenders_addons as tfra
```

### Compatility with Tensorflow
TensorFlow C++ APIs are not stable and thus we can only guarantee compatibility with the 
version TensorFlow Recommenders-Addons(TFRA) was built against. It is possible TFRA will work with 
multiple versions of TensorFlow, but there is also a chance for segmentation faults or other problematic 
crashes. Warnings will be emitted when loading a custom op if your TensorFlow version does not match 
what it was built against.

Additionally, custom ops registration does not have a stable ABI interface so it is 
required that users have a compatible installation of TensorFlow even if the versions 
match what we had built against. A simplification of this is that **TensorFlow Recommenders-Addons 
custom ops will work with `pip`-installed TensorFlow** but will have issues when TensorFlow 
is compiled differently. A typical example of this would be `conda`-installed TensorFlow.
[RFC #133](https://github.com/tensorflow/community/pull/133) aims to fix this.


#### Compatibility Matrix
| TensorFlow Recommenders-Addons | TensorFlow | Compiler  |
|:----------------------- |:---- |:---------|
| tensorflow-recommenders-addons-0.1.0 | 2.4.1  | GCC 7.3.1 |

**NOTICE**：The release packages have strict version binding relationship with TensorFlow. 
If you need to work with other versions of TensorFlow, we recommend you installing from source.


#### Installing from Source
You can also install from source. This requires the [Bazel](https://bazel.build/) build system (version >= 1.0.0).

```
git clone https://github.com/tensorflow/recommenders-addons.git
cd recommenders-addons

# This script links project with TensorFlow dependency
python3 ./configure.py

bazel build --enable_runfiles build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_recommenders_addons-*.whl
```

### Compatility with Tensorflow Serving

#### Compatibility Matrix
| TensorFlow Recommenders-Addons | TensorFlow | Serving | Compiler  |
|:----------------------- |:---- |:---- |:---------|
| tensorflow-recommenders-addons-0.1.0 | 2.4.1  | 2.4.0  | GCC 7.3.1 |

#### Serving TensorFlow models with custom ops
Reference documents: https://www.tensorflow.org/tfx/serving/custom_op

TFRA modification([`tensorflow_recommenders_addons.bzl`](tensorflow_recommenders_addons/tensorflow_recommenders_addons.bzl)):
```
deps = deps + [
        # "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ]

native.cc_library(
        name = name,
        srcs = srcs,
        copts = copts,
        alwayslink = 1,
        features = select({
            "//tensorflow_recommenders_addons:windows": ["windows_export_all_symbols"],
            "//conditions:default": [],
        }),
        deps = deps,
        **kwargs
    )
```
Tensorflow Serving modification(**model_servers/BUILD**):
```
SUPPORTED_TENSORFLOW_OPS = if_v2([]) + if_not_v2([
    "@org_tensorflow//tensorflow/contrib:contrib_kernels",
    "@org_tensorflow//tensorflow/contrib:contrib_ops_op_lib",
]) + [
    "@org_tensorflow_text//tensorflow_text:ops_lib",
    "//tensorflow_recommenders_addons/dynamic_embedding/core:_cuckoo_hashtable_ops.so",
]
```

## Contributing

TensorFlow Recommenders-Addons is a community-led open source project. As such,
the project depends on public contributions, bug fixes, and documentation. This
project adheres to TensorFlow's Code of Conduct. 

Please follow up the [contributing guide](CONTRIBUTING.md) for more details.

## Community

* SIG Recommenders mailing list:
[recommenders@tensorflow.org](https://groups.google.com/a/tensorflow.org/g/recommenders)

## Acknowledgment
We are very grateful to the maintainers of [tensorflow/addons](https://github.com/tensorflow/addons) for borrowing a lot of code from [tensorflow/addons](https://github.com/tensorflow/addons) to build our workflow and documentation system.

## Licence
Apache License 2.0

