licenses(["notice"])  # Apache 2.0

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda", "if_cuda_is_configured")

package(default_visibility = ["//visibility:public"])

load("//tensorflow_recommenders_addons:tensorflow_recommenders_addons.bzl", "custom_op_library", "if_cuda_for_tf_serving")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_tf//:build_defs.bzl", "FOR_TF_SERVING")

custom_op_library(
    name = "_cuckoo_hashtable_ops.so",
    srcs = [
        "kernels/cuckoo_hashtable_op.h",
        "kernels/cuckoo_hashtable_op.cc",
        "ops/cuckoo_hashtable_ops.cc",
    ] + glob(["kernels/lookup_impl/lookup_table_op_cpu*"]),
    cuda_deps = if_cuda_for_tf_serving(
        ["//tensorflow_recommenders_addons/dynamic_embedding/core/lib/nvhash:nvhashtable"],
        [],
        FOR_TF_SERVING,
    ),
    cuda_srcs = if_cuda([
        "kernels/cuckoo_hashtable_op_gpu.h",
        "kernels/cuckoo_hashtable_op_gpu.cu.cc",
    ] + glob(["kernels/lookup_impl/lookup_table_op_gpu*"])),
    deps = ["//tensorflow_recommenders_addons/dynamic_embedding/core/lib/cuckoo:cuckoohash"],
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
    ],
    cuda_srcs = [
        "kernels/segment_reduction_ops.h",
        "kernels/segment_reduction_ops_gpu.cu.cc",
        "kernels/sparse_fill_empty_rows_op.h",
        "kernels/sparse_fill_empty_rows_op.cu.cc",
        "kernels/sparse_reshape_op.h",
        "kernels/sparse_reshape_op.cu.cc",
    ],
)
