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

## Subpackages

* [tfra.dynamic_embedding](docs/api_docs/tfra/dynamic_embedding.md)
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

By default, CPU version will be installed. To install GPU version, run the following:
```
pip install tensorflow-recommenders-addons-gpu
```

To use TensorFlow Recommenders-Addons:

```python
import tensorflow as tf
import tensorflow_recommenders_addons as tfra
```

### Compatibility with Tensorflow
TensorFlow C++ APIs are not stable and thus we can only guarantee compatibility with the 
version TensorFlow Recommenders-Addons(TFRA) was built against. It is possible TFRA will work with 
multiple versions of TensorFlow, but there is also a chance for segmentation faults or other problematic 
crashes. Warnings will be emitted if your TensorFlow version does not match what it was built against.

Additionally, TFRA custom ops registration does not have a stable ABI interface so it is 
required that users have a compatible installation of TensorFlow even if the versions 
match what we had built against. A simplification of this is that **TensorFlow Recommenders-Addons 
custom ops will work with `pip`-installed TensorFlow** but will have issues when TensorFlow 
is compiled differently. A typical example of this would be `conda`-installed TensorFlow.
[RFC #133](https://github.com/tensorflow/community/pull/133) aims to fix this.


#### Compatibility Matrix
*GPU is supported from version `0.2.0`*

| TFRA | TensorFlow | Compiler  | CUDA | CUDNN | Compute Capability |
|:----------------------- |:---- |:---------| :------------ | :---- | :------------ |
| 0.2.0 | 2.4.1  | GCC 7.3.1 | 11.0 | 8.0 | 6.0, 6.1, 7.0, 7.5, 8.0 |
| 0.2.0 | 1.15.2  | GCC 7.3.1 | 10.0 | 7.6 | 6.0, 6.1, 7.0, 7.5 |
| 0.1.0 | 2.4.1  | GCC 7.3.1 | - | - | - |

Check [nvidia-support-matrix](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html) for more details.

**NOTICE**
- The release packages have strict version binding relationship with TensorFlow. 
- Due to the significant changes in the Tensorflow API, we can only ensure compatibility with TF1.15.2 on CPU & GPU, 
  but **there are no official releases**, you can only get it through compiling by the following:
```shell
PY_VERSION="3.6" \
TF_VERSION="1.15.2" \
TF_NEED_CUDA=1 \
TF_CUDA_VERSION=10.0 \
TF_CUDNN_VERSION=7.6 \
sh .github/workflows/make_wheel_Linux.sh

# .whl file will be created in ./wheelhouse/
```

- If you need to work with TensorFlow 1.14.x or older version, we suggest you give up,
but maybe this doc can help you : [Extract headers from TensorFlow compiling dirctory](./build_deps/tf_header/README.md).
At the same time, we find some OPs used by TRFA have better performance, so we highly recommend you update TensorFlow to 2.x.

#### Installing from Source
##### CPU Only
You can also install from source. This requires the [Bazel](https://bazel.build/) build system (version == 3.1.0).
Please install a TensorFlow on your compiling machine, The compiler needs to know the version of Tensorflow and 
its headers according to the installed TensorFlow. 

```
export TF_VERSION=["2.4.1", "1.15.2"]
pip install tensorflow[-gpu]==$TF_VERSION

git clone https://github.com/tensorflow/recommenders-addons.git
cd recommenders-addons

# This script links project with TensorFlow dependency
python configure.py

bazel build --enable_runfiles build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_recommenders_addons-*.whl
```

##### GPU Support
Only `TF_NEED_CUDA=1` is required and other environment variables are optional:
```shell
PY_VERSION="3.6" \
TF_NEED_CUDA=1 \
TF_CUDA_VERSION=11.0 \
TF_CUDNN_VERSION=8.0 \
CUDA_TOOLKIT_PATH="/usr/local/cuda" \
CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu" \
python configure.py
```
And then build the pip package and install:
```shell
bazel build --enable_runfiles build_pip_pkg
TF_NEED_CUDA=1 bazel-bin/build_pip_pkg artifacts
pip install artifacts/tensorflow_recommenders_addons_gpu-*.whl
```

##### Data Type Matrix for `tfra.dynamic_embedding.Variable` 
|  Values \\ Keys  | int64  | int32 | string |
|:----:|:----:|:----:|:----:| 
| float  | CPU, GPU | CPU | CPU |
| half  | CPU, GPU | - | CPU |
| int32  | CPU, GPU | CPU | CPU |
| int8  | CPU, GPU | - | CPU |
| int64  | CPU | - | CPU |
| double  | CPU, CPU | CPU | CPU |
| bool  | - | - | CPU |
| string  | CPU | - | - |

##### To use GPU by `tfra.dynamic_embedding.Variable`
The `tfra.dynamic_embedding.Variable` will ignore the device placement mechanism of TensorFlow, 
you should specify the `devices` onto GPUs explictly for it.
```python
import tensorflow as tf
import tensorflow_recommenders_addons as tfra

de = tfra.dynamic_embedding.get_variable("VariableOnGpu",
                                         devices=["/job:ps/task:0/GPU:0", ],
                                         # ...
                                         )
```

**Usage restrictions on GPU**
- Only work on Nvidia GPU with cuda compute capability 6.0 or higher.
- Considering the size of the .whl file, currently `dim` only supports less than or equal to 200, if you need longer `dim`, please submit an issue.
- Only `dynamic_embedding` APIs and relative OPs support running on GPU.
- For GPU HashTables manage GPU memory independently, TensorFlow should be configured to allow GPU memory growth by the following:
```python
sess_config.gpu_options.allow_growth = True
```

### Compatibility with Tensorflow Serving

#### Compatibility Matrix
| TFRA | TensorFlow | Serving | Compiler  | CUDA | CUDNN | Compute Capability |
|:----- |:---- |:---- |:---------| :------------ | :---- | :------------ |
| 0.2.0 | 2.4.1  | 2.4.0  | GCC 7.3.1 | 11.0 | 8.0 | 6.0, 6.1, 7.0, 7.5, 8.0 |
| 0.2.0 | 1.15.2  | 1.15.0  | GCC 7.3.1 | 10.0 | 7.6 | 6.0, 6.1, 7.0, 7.5 |
| 0.1.0 | 2.4.1  | 2.4.0  | GCC 7.3.1 | - | - | - |

**NOTICE**：Reference documents: https://www.tensorflow.org/tfx/serving/custom_op

#### CPU or GPU Serving TensorFlow models with custom ops
When compiling, set the environment variable:
```
export FOR_TF_SERVING = "1"
```
Tensorflow Serving modification(**model_servers/BUILD**):
```
SUPPORTED_TENSORFLOW_OPS = if_v2([]) + if_not_v2([
    "@org_tensorflow//tensorflow/contrib:contrib_kernels",
    "@org_tensorflow//tensorflow/contrib:contrib_ops_op_lib",
]) + [
    "@org_tensorflow_text//tensorflow_text:ops_lib",
    "//tensorflow_recommenders_addons/dynamic_embedding/core:_cuckoo_hashtable_ops.so",
    "//tensorflow_recommenders_addons/dynamic_embedding/core:_math_ops.so",
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

