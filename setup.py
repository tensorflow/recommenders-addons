# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorFlow Recommenders Addons.

TensorFlow Recommenders Addons are a collection of projects related to
large-scale recommendation systems built upon TensorFlow. They are contributed
and maintained by the community. Those contributions will be complementary to
TensorFlow Core and TensorFlow Recommenders etc.
"""

import os
from pathlib import Path
import sys
import platform

from datetime import datetime
from setuptools import find_namespace_packages
from setuptools import setup
from setuptools.dist import Distribution
from setuptools import Extension
try:
  from packaging.version import Version
except:  # make it compatible for python 3.7
  from distutils.version import LooseVersion as Version

DOCLINES = __doc__.split("\n")


def get_last_commit_time() -> str:
  string_time = os.getenv("NIGHTLY_TIME").replace('"', "")
  return datetime.strptime(string_time,
                           "%Y-%m-%dT%H:%M:%SZ").strftime("%Y%m%d%H%M%S")


def get_project_name_version():
  # Version
  version = {}
  base_dir = os.path.dirname(os.path.abspath(__file__))
  with open(
      os.path.join(base_dir, "tensorflow_recommenders_addons",
                   "version.py")) as fp:
    exec(fp.read(), version)

  project_name = "tensorflow-recommenders-addons"
  version["tf_project_name"] = "tensorflow"
  if os.getenv("TF_NEED_CUDA", "0") == "1":
    project_name = project_name + "-gpu"

  if "--nightly" in sys.argv:
    project_name = "tfra-nightly"
    if os.getenv("TF_NEED_CUDA", "0") == "1":
      project_name = project_name + "-gpu"
    version["__version__"] += get_last_commit_time()
    sys.argv.remove("--nightly")

  return project_name, version


def get_ext_modules():
  ext_modules = []
  if "--platlib-patch" in sys.argv:
    if sys.platform.startswith("linux"):
      # Manylinux2010 requires a patch for platlib
      ext_modules = [Extension("_foo", ["stub.cc"])]
    sys.argv.remove("--platlib-patch")
  return ext_modules


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return True


project_name, version = get_project_name_version()
min_tf_version = version["MIN_TF_VERSION"]
max_tf_version = version["MAX_TF_VERSION"]
tf_project_name = version["tf_project_name"]
setup(
    name=project_name,
    version=version["__version__"],
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    author="Google Inc.",
    author_email="opensource@google.com",
    packages=find_namespace_packages(
        include=['tensorflow_recommenders_addons*']),
    ext_modules=get_ext_modules(),
    install_requires=Path("requirements.txt").read_text().splitlines() +
    ["{}>={},<={}".format(tf_project_name, min_tf_version, max_tf_version)],
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    license="Apache 2.0",
    keywords="tensorflow recommenders addons machine learning",
)
