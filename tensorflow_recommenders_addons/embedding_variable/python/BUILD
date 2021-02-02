# Placeholder for internal Python library.

package(
    default_visibility = [
        "//tensorflow_recommenders_addons:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "embedding_variable",
    srcs = glob(["*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_recommenders_addons/embedding_variable/python/ops:embedding_variable_ops",
        "//tensorflow_recommenders_addons/embedding_variable/python/optimizers:optimizer_ops",
    ],
)
