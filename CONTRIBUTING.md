# Contributing

TensorFlow Recommender Addons are a collection of projects related to
large-scale recommendation systems built upon TensorFlow. They are contributed
and maintained by the community. Those contributions will be complementary to
TensorFlow Core and TensorFlow Recommenders etc.

## Maintainership

We adopt proxy maintainership as in [TensorFlow recommenders-addons](https://github.com/tensorflow/recommenders-addons):

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

### Pull Requests
We welcome contributions via pull requests.
Before sending out a pull request, we recommend that you open an issue and
discuss your proposed change. Some changes may require a design review.
All submissions require review by project owners or TensorFlow Recommenders SIG
leads.

**NOTE**:
If your PR cannot be mereged, and system indicate you like "Merging is blocked, 
The base branch requres all commits to be signed'
You have to configure your git and GPG key to sign your commit. [Sign your commit with GPG key](https://docs.github.com/en/github/authenticating-to-github/managing-commit-signature-verification/about-commit-signature-verification#gpg-commit-signature-verification) .

### Design Review
A new project in this repository or a significant change to an existing project
requires a design review. We recommend that you discuss your idea in the mailing
list (recommenders@tensorflow.org) before moving forward.

The centerpiece of a design review is a design doc which needs to include the following:
* Motivation of the change
* High-level design
* Detailed changes and related changes in TensorFlow
* [Optional] other alternatives that have been considered
* [Optional] testing plan
* [Optional] maintenance plan

The author needs to send out the design doc via a pull request. Project owners or
TensorFlow SIG Recommenders leads will discuss proposals in a monthly meeting
or an ad-hoc design review meeting. After a proposal is approved, the author
could then start contributing the implementation.

### Coding Style
We require all contribution conforms to [TensorFlow Style Guide](https://www.tensorflow.org/community/contribute/code_style#tensorflow_conventions_and_special_uses).
See our [Style Guide](STYLE_GUIDE.md) for more details.

### Additional Requirements
In addition to the above requirements, contribution also needs to meet the following criteria:
* The functionality is not otherwise available in TensorFlow.
* It has to be compatible with TensorFlow 2.6.3 and 2.15.1.
* The change needs to include unit tests and integration tests if any.
* Each project needs to provide documentation for when and how to use it.

## Community

* TensorFlow Recommenders code (github.com/tensorflow/recommenders-addons)
* SIG Recommenders mailing list (https://groups.google.com/a/tensorflow.org/g/recommenders)

## Licence
Apache License 2.0

