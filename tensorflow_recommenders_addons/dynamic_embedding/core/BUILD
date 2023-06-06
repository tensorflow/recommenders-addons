licenses(["notice"])  # Apache 2.0

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda", "if_cuda_is_configured")

package(default_visibility = ["//visibility:public"])

load("//tensorflow_recommenders_addons:tensorflow_recommenders_addons.bzl", "custom_table_library", "custom_op_library", "if_cuda_for_tf_serving")
load("//tensorflow_recommenders_addons:dynamic_embedding/lookup_table.bzl", "build_lookup_table_library")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_tf//:build_defs.bzl", "FOR_TF_SERVING")

custom_op_library(
    name = "_cuckoo_hashtable_ops.so",
    srcs = [
        "kernels/cuckoo_hashtable_op.h",
        "kernels/cuckoo_hashtable_op.cc",
        "ops/cuckoo_hashtable_ops.cc",
        "utils/utils.h",
        "utils/types.h",
    ] + glob(["kernels/lookup_impl/lookup_table_op_cpu*"]),
    cuda_deps = if_cuda_for_tf_serving(
        ["//tensorflow_recommenders_addons/dynamic_embedding/core/lib/nvhash:nvhashtable"],
        [],
        FOR_TF_SERVING,
    ),
    cuda_srcs = if_cuda([
        "kernels/cuckoo_hashtable_op.h",
        "kernels/cuckoo_hashtable_op_gpu.h",
        "kernels/cuckoo_hashtable_op_gpu.cu.cc",
        "utils/utils.h",
        "utils/types.h",
    ] + glob(["kernels/lookup_impl/lookup_table_op_gpu*"])),
    deps = [
        "//tensorflow_recommenders_addons/dynamic_embedding/core/lib/cuckoo:cuckoohash",
    ],
)

custom_op_library(
    name = "_redis_table_ops.so",
    srcs = [
        "kernels/redis_impl/json.cc",
        "kernels/redis_impl/json.h",
        "kernels/redis_impl/md5.cc",
        "kernels/redis_impl/md5.h",
        "kernels/redis_impl/redis_cluster_connection_pool.hpp",
        "kernels/redis_impl/redis_connection_pool.hpp",
        "kernels/redis_impl/redis_connection_util.hpp",
        "kernels/redis_impl/redis_slots_tab.h",
        "kernels/redis_impl/redis_table_op_util.hpp",
        "kernels/redis_impl/thread_pool.h",
        "kernels/redis_table_op.cc",
        "kernels/redis_table_op.h",
        "ops/redis_table_ops.cc",
        "utils/types.h",
        "utils/utils.h",
    ],
    copts = [
        "-pthread",
        "-Ofast",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:linux_x86_64": [
            "-lpthread",
            "-lrt",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "@hiredis",
        "@redis-plus-plus//:redis-plus-plus",
    ],
)

custom_op_library(
    name = "_math_ops.so",
    srcs = [
        "kernels/segment_reduction_ops.h",
        "kernels/segment_reduction_ops_impl.cc",
        "kernels/segment_reduction_ops_impl.h",
        "kernels/sparse_fill_empty_rows_op.cc",
        "kernels/sparse_fill_empty_rows_op.h",
        "kernels/sparse_reshape_op.cc",
        "kernels/sparse_reshape_op.h",
        "ops/math_ops.cc",
        "utils/utils.h",
    ],
    cuda_deps = if_cuda_for_tf_serving(
        ["//tensorflow_recommenders_addons/dynamic_embedding/core/lib/nvhash:nvhashtable"],
        [],
        FOR_TF_SERVING,
    ),
    cuda_srcs = [
        "kernels/segment_reduction_ops.h",
        "kernels/segment_reduction_ops_gpu.cu.cc",
        "kernels/sparse_fill_empty_rows_op.h",
        "kernels/sparse_fill_empty_rows_op.cu.cc",
        "kernels/sparse_reshape_op.h",
        "kernels/sparse_reshape_op.cu.cc",
    ],
)

build_lookup_table_library(
    name = "tfra_register",
    srcs = [
        "kernels/lookup_table_op_registry.cc",
        "kernels/lookup_table_types.cc",
        "kernels/lookup_registration.h",
        "utils/types.h",
        "utils/utils.h",
    ],
    hdrs = [
        "kernels/lookup_table_types.h",
        "kernels/lookup_table_interface.h",
        "kernels/lookup_table_op_registry.h",
    ],
    copts = [
        "-Ofast",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:linux_x86_64": [
            "-lrt",
        ],
        "//conditions:default": [],
    }),
)

custom_op_library(
    name = "_lookup_table_ops.so",
    srcs = [
        "kernels/lookup_table.hpp",
        "kernels/lookup_table_cpu_ops.cc",
        "ops/lookup_table_ops.cc",
        "utils/types.h",
        "utils/utils.h",
    ],
    copts = [
        "-Ofast",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:linux_x86_64": [
            "-lrt",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":tfra_register"
    ]
)

custom_table_library(
    name = "_std_redis_table_ops.so",
    srcs = [
        "utils/utils.h",
        "utils/types.h",
        "kernels/uniform_redis_impl/json.cc",
        "kernels/uniform_redis_impl/json.h",
        "kernels/uniform_redis_impl/md5.cc",
        "kernels/uniform_redis_impl/md5.h",
        "kernels/uniform_redis_impl/redis_cluster_connection_pool.hpp",
        "kernels/uniform_redis_impl/redis_connection_pool.hpp",
        "kernels/uniform_redis_impl/redis_connection_util.hpp",
        "kernels/uniform_redis_impl/redis_slots_tab.h",
        "kernels/uniform_redis_impl/redis_table_op_util.hpp",
        "kernels/uniform_redis_impl/thread_pool.h",
        "kernels/std_redis_table_op.cc",
    ],
    copts = [
        "-pthread",
        "-Ofast",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:linux_x86_64": [
            "-lpthread",
            "-lrt",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "@hiredis",
        "@redis-plus-plus//:redis-plus-plus",
        ":tfra_register"
    ],
)

# custom_op_library(
#     name = "_std_cuckoo_hashtable_ops.so",
#     srcs = [
#         "kernels/std_cuckoo_hashtable_op.cc",
#         "utils/utils.h",
#         "utils/types.h",
#         "kernels/lookup_table_interface.h",
#         "kernels/lookup_table_op_registry.h",
#     ] + glob(["kernels/lookup_impl/lookup_table_op_cpu*"]),
#     copts = [
#         "-g",
#     ],
#     deps = [
#         "//tensorflow_recommenders_addons/dynamic_embedding/core/lib/cuckoo:cuckoohash",
#         ":tfra_register"
#     ],
# )

# custom_op_library(
#     name = "_std_cuckoo_hashtable_ops_gpu.so",
#     cuda_deps = if_cuda_for_tf_serving(
#         ["//tensorflow_recommenders_addons/dynamic_embedding/core/lib/nvhash:nvhashtable"],
#         [],
#         FOR_TF_SERVING,
#     ),
#     cuda_srcs = if_cuda([
#         "kernels/std_cuckoo_hashtable_op_gpu.cu.cc",
#         "utils/utils.h",
#         "utils/types.h",
#     ] + glob(["kernels/lookup_impl/lookup_table_op_gpu*"])),
#     copts = [
#         # "-Ofast",
#         "-g",
#     ],
#     deps = [
#         "//tensorflow_recommenders_addons/dynamic_embedding/core/lib/cuckoo:cuckoohash",
#         ":tfra_register"
#     ]
# )