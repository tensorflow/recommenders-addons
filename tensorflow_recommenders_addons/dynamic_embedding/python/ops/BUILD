# Placeholder for internal Python library.
# Placeholder for internal Python test library.

package(
    default_visibility = [
        "//tensorflow_recommenders_addons:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "dynamic_embedding",
    srcs = glob(["*.py"]),
    srcs_version = "PY3",
    data = [
        "//tensorflow_recommenders_addons/dynamic_embedding/core:_cuckoo_hashtable_ops.so",
    ],
)

