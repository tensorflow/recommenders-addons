# Placeholder for internal Python library.
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

package(
    default_visibility = [
        "//tensorflow_recommenders_addons:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "dynamic_embedding",
    srcs = glob(["*.py"]),
    data = if_cuda(
        [
            "//tensorflow_recommenders_addons/dynamic_embedding/core:_cuckoo_hashtable_ops.so",
            "//tensorflow_recommenders_addons/dynamic_embedding/core:_data_flow_ops.so",
            "//tensorflow_recommenders_addons/dynamic_embedding/core:_hkv_ops.so",
            "//tensorflow_recommenders_addons/dynamic_embedding/core:_math_ops.so",
            "//tensorflow_recommenders_addons/dynamic_embedding/core:_redis_table_ops.so",
        ],
        [
            "//tensorflow_recommenders_addons/dynamic_embedding/core:_cuckoo_hashtable_ops.so",
            "//tensorflow_recommenders_addons/dynamic_embedding/core:_data_flow_ops.so",
            "//tensorflow_recommenders_addons/dynamic_embedding/core:_math_ops.so",
            "//tensorflow_recommenders_addons/dynamic_embedding/core:_redis_table_ops.so",
        ],
    ),
    srcs_version = "PY2AND3",
)
