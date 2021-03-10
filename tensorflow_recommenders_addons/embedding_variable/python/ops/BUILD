# Placeholder for internal Python library.

package(
    default_visibility = [
        "//tensorflow_recommenders_addons:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "embedding_variable_ops",
    srcs = glob(["*.py"]),
    data = [
        "//tensorflow_recommenders_addons/embedding_variable/core:_ev_ops.so",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "embedding_variable_ops_test",
    srcs = ["embedding_variable_ops_test.py"],
    data = [
        "//tensorflow_recommenders_addons/embedding_variable/core:_ev_ops.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_recommenders_addons",
    ],
)
