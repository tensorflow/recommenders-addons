# TensorFlow Recommenders Addons

TensorFlow Recommender Addons are a collection of projects related to
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

## Maintainership

We adopt proxy maintainership as in [TensorFlow addons](https://github.com/tensorflow/addons):

*Projects and subpackages are compartmentalized and each is maintained by those
with expertise and vested interest in that component.*

*Subpackage maintainership will only be granted after substantial contribution
has been made in order to limit the number of users with write permission.
Contributions can come in the form of issue closings, bug fixes, documentation,
new code, or optimizing existing code. Submodule maintainership can be granted
with a lower barrier for entry as this will not include write permissions to
the repo.*

## Contributing

TensorFlow Recommenders Addons is a community-led open source project. As such,
the project depends on public contributions, bug fixes, and documentation. This
project adheres to TensorFlow's Code of Conduct. 

Please follow up the [contributing guide](CONTRIBUTING.md) for more details.

## Community

* SIG Recommenders mailing list:
[recommenders@tensorflow.org](https://groups.google.com/a/tensorflow.org/g/recommenders)

## Licence
Apache License 2.0

