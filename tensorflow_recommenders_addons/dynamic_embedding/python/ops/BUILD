# Placeholder for internal Python library.

package(
    default_visibility = [
        "//tensorflow_recommenders_addons:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "dynamic_embedding",
    srcs = glob(["*.py"]),
    data = [
        "//tensorflow_recommenders_addons/dynamic_embedding/core:_cuckoo_hashtable_ops.so",
        "//tensorflow_recommenders_addons/dynamic_embedding/core:_math_ops.so",
        "//tensorflow_recommenders_addons/dynamic_embedding/core:_redis_table_ops.so",
        "//tensorflow_recommenders_addons/dynamic_embedding/core:tfra_register",
        "//tensorflow_recommenders_addons/dynamic_embedding/core:_lookup_table_ops.so",
        "//tensorflow_recommenders_addons/dynamic_embedding/core:_std_redis_table_ops.so",
        "//tensorflow_recommenders_addons/dynamic_embedding/core:_std_cuckoo_hashtable_ops.so",
        "//tensorflow_recommenders_addons/dynamic_embedding/core:_std_cuckoo_hashtable_ops_gpu.so",
    ],
    srcs_version = "PY2AND3",
)
