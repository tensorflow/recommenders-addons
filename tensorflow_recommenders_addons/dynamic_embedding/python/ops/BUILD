# Placeholder for internal Python library.
# Placeholder for internal Python test library.

package(
    default_visibility = [
        "//tensorflow_recommenders_addons:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "dynamice_embedding",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":dynamic_embedding_ops",
        ":tf_hacker",
    ],
)

py_library(
    name = "dynamic_embedding_ops",
    srcs = ["dynamic_embedding_ops.py",
            "dynamic_embedding_optimizer.py",
            "trainable_wrapper.py"],
    srcs_version = "PY3",
    deps = [
        # py/tensorflow dep,
    ],
)

py_library(
    name = "tf_hacker",
    srcs = ["tf_hacker.py"],
    srcs_version = "PY3",
    deps = [
        # py/tensorflow dep,
    ],
)

# Tests
py_test(
    name = "dynamic_embedding_ops_test",
    srcs = ["tests/dynamic_embedding_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dynamic_embedding_ops",
    ],
)

py_test(
    name = "dynamic_embedding_ops_trainable_test",
    srcs = ["tests/dynamic_embedding_ops_trainable_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dynamic_embedding_ops",
    ],
)

