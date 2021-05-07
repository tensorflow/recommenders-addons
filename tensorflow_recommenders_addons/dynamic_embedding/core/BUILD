licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_recommenders_addons:tensorflow_recommenders_addons.bzl", "custom_op_library")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

custom_op_library(
    name = "_cuckoo_hashtable_ops.so",
    srcs = [
        "kernels/cuckoo_hashtable_op.cc",
        "kernels/cuckoo_hashtable_op.h",
        "ops/cuckoo_hashtable_ops.cc",
    ] + glob(["kernels/lookup_impl/lookup_table_op_cpu*"]),
    deps = ["//tensorflow_recommenders_addons/dynamic_embedding/core/lib/cuckoo:cuckoohash"],
)

custom_op_library(
    name = "_segment_reduction_ops.so",
    srcs = [
        "kernels/segment_reduction_ops.h",
        "kernels/segment_reduction_ops_impl.cc",
        "kernels/segment_reduction_ops_impl.h",
        "ops/math_ops.cc",
    ],
    cuda_srcs = [
        "kernels/segment_reduction_ops.h",
        "kernels/segment_reduction_ops_gpu.cu.cc",
    ],
)
