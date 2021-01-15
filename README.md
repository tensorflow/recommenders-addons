# TensorFlow Recommenders Addons
-----------------

[![PyPI Status Badge](https://badge.fury.io/py/tensorflow-recommenders-addons.svg)](https://pypi.org/project/tensorflow-recommenders-addons/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tensorflow-recommenders-addons)](https://pypi.org/project/tensorflow-recommenders-addons/)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/recommenders-addons/api_docs/python/tfra)
[![Gitter chat](https://img.shields.io/badge/chat-on%20gitter-46bc99.svg)](https://gitter.im/tensorflow/sig-recommenders-addons)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


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
* [RFC: Sparse Domain Isolation](rfcs/20200424-sparse-domain-isolation.md)

## Recommender-Addons Subpackages

* [tfra.dynamic_embedding](https://www.tensorflow.org/addons/api_docs/python/tfra/dynamic_embdding)
   
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

### Python Op Compatility
TensorFlow Recommenders-Addons is actively working towards forward compatibility with TensorFlow 2.x. 
However, there are still a few private API uses within the repository so at the moment 
we can only guarantee compatibility with the TensorFlow versions which it was tested against. 
Warnings will be emitted when importing `tensorflow_recommenders_addons` if your TensorFlow version  
does not match what it was tested against.

#### Python Op Compatibility Matrix
| TensorFlow Recommenders-Addons | TensorFlow | Python  |
|:----------------------- |:---|:---------- |
| tfra-nightly | 2.3 | 3.6, 3.7, 3.8 | 
| tensorflow-recommenders-addons-0.1.0 | 2.3 |3.6, 3.7, 3.8 |

### C++ Custom Op Compatibility
TensorFlow C++ APIs are not stable and thus we can only guarantee compatibility with the 
version TensorFlow Recommenders-Addons was built against. It is possible custom ops will work with 
multiple versions of TensorFlow, but there is also a chance for segmentation faults or other problematic 
crashes. Warnings will be emitted when loading a custom op if your TensorFlow version does not match 
what it was built against.

Additionally, custom ops registration does not have a stable ABI interface so it is 
required that users have a compatible installation of TensorFlow even if the versions 
match what we had built against. A simplification of this is that **TensorFlow Recommenders-Addons 
custom ops will work with `pip`-installed TensorFlow** but will have issues when TensorFlow 
is compiled differently. A typical example of this would be `conda`-installed TensorFlow.
[RFC #133](https://github.com/tensorflow/community/pull/133) aims to fix this.


#### C++ Custom Op Compatibility Matrix
| TensorFlow Recommenders-Addons | TensorFlow | Compiler  | cuDNN | CUDA | 
|:----------------------- |:---- |:---------|:---------|:---------|
| tfra-nightly | 2.3 | GCC 7.3.1 | 7.6 | 10.1 |
| tensorflow-recommenders-addons-0.1.0 | 2.3  | GCC 7.3.1 | 7.6 | 10.1 |


#### Nightly Builds
There are also nightly builds of TensorFlow Recommenders-Addons under the pip package
`tfra-nightly`, which is built against **the latest stable version of TensorFlow**. Nightly builds
include newer features, but may be less stable than the versioned releases. Contrary to 
what the name implies, nightly builds are not released every night, but at every commit 
of the master branch. `0.1.0.dev20201225174950` means that the commit time was 
2020/12/25 at 17:49:50 Coordinated Universal Time.

```
pip install tfra-nightly
```

#### Installing from Source
You can also install from source. This requires the [Bazel](https://bazel.build/) build system (version >= 1.0.0).

##### CPU Custom Ops
```
git clone https://github.com/tensorflow/recommenders-addons.git
cd recommenders-addons

# This script links project with TensorFlow dependency
python3 ./configure.py

bazel build --enable_runfiles build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_recommenders_addons-*.whl
```

## Tutorials
See [`docs/tutorials/`](docs/tutorials/)
for end-to-end examples of various addons.

## Contributing

TensorFlow Recommenders-Addons is a community-led open source project. As such,
the project depends on public contributions, bug fixes, and documentation. This
project adheres to TensorFlow's Code of Conduct. 

Please follow up the [contributing guide](CONTRIBUTING.md) for more details.

## Community

* SIG Recommenders mailing list:
[recommenders@tensorflow.org](https://groups.google.com/a/tensorflow.org/g/recommenders)

## Licence
Apache License 2.0

