licenses(["notice"])  # Apache 2.0

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda", "if_cuda_is_configured")

package(default_visibility = ["//visibility:public"])

load("//tensorflow_recommenders_addons:tensorflow_recommenders_addons.bzl", "custom_cuda_op_library", "custom_op_library", "if_cuda_for_tf_serving")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_tf//:build_defs.bzl", "FOR_TF_SERVING")

custom_op_library(
    name = "_cuckoo_hashtable_ops.so",
    srcs = [
        "kernels/cuckoo_hashtable_op.cc",
        "kernels/cuckoo_hashtable_op.h",
        "ops/cuckoo_hashtable_ops.cc",
        "utils/types.h",
        "utils/utils.h",
    ] + glob(["kernels/lookup_impl/lookup_table_op_cpu*"]),
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

# _math_ops.so has been removed since TF2 performance is good enough.
config_setting(
    name = "build_math_op",
    values = {
        "define": "BUILD_MATH_OP=1",
    },
)

custom_op_library(
    name = "_math_ops.so",
    srcs = select({
        ":build_math_op": [
            "kernels/segment_reduction_ops.h",
            "kernels/segment_reduction_ops_impl.cc",
            "kernels/segment_reduction_ops_impl.h",
            "ops/math_ops.cc",
            "utils/utils.h",
        ],
        "//conditions:default": [],
    }),
    cuda_srcs = select({
        ":build_math_op": [
            "kernels/segment_reduction_ops.h",
            "kernels/segment_reduction_ops_gpu.cu.cc",
        ],
        "//conditions:default": [],
    }),
)

# _data_flow_ops.so has been removed since TF2.9 already support int32/int64 GPU op.
config_setting(
    name = "build_data_flow_op",
    values = {
        "define": "BUILD_DATA_FLOW_OP=1",
    },
)

custom_op_library(
    name = "_data_flow_ops.so",
    srcs = select({
        ":build_data_flow_op": [
            "kernels/dynamic_partition_op.cc",
            "kernels/dynamic_stitch_op.cc",
            "ops/data_flow_ops.cc",
            "utils/utils.h",
        ],
        "//conditions:default": [],
    }),
    cuda_srcs = select({
        ":build_data_flow_op": [
            "kernels/fill_functor.cu.cc",
            "kernels/dynamic_partition_op_gpu.cu.cc",
            "kernels/dynamic_stitch_op_gpu.cu.cc",
            "utils/utils.h",
        ],
        "//conditions:default": [],
    }),
)

# TODO: Add hkv targets.
custom_cuda_op_library(
    name = "_hkv_ops.so",
    srcs = [
        "kernels/cuckoo_hashtable_op.h",
        "kernels/hkv_hashtable_op.cc",
        "ops/hkv_hashtable_ops.cc",
        "utils/utils.h",
    ] + glob(["kernels/lookup_impl/lookup_table_op_cpu*"]),
    copts = [
        "-Ofast",
    ],
    cuda_deps = if_cuda_for_tf_serving(
        ["@hkv//:hkv"],
        [],
        FOR_TF_SERVING,
    ),
    cuda_srcs = if_cuda([
        "utils/utils.h",
        "utils/types.h",
        "kernels/cuckoo_hashtable_op_gpu.h",
        "kernels/hkv_hashtable_op_gpu.cu.cc",
    ]) + glob(["kernels/lookup_impl/lookup_table_op_hkv*"]),
    deps = [
        "//tensorflow_recommenders_addons/dynamic_embedding/core/lib/cuckoo:cuckoohash",
    ],
)
