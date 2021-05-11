licenses(["notice"])  # Apache 2.0

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda", "if_cuda_is_configured")

package(default_visibility = ["//visibility:public"])

load("//tensorflow_recommenders_addons:tensorflow_recommenders_addons.bzl", "custom_op_library")

custom_op_library(
    name = "_cuckoo_hashtable_ops.so",
    srcs = [
        "kernels/cuckoo_hashtable_op.h",
        "kernels/cuckoo_hashtable_op.cc",
        "ops/cuckoo_hashtable_ops.cc",
    ] + glob(["kernels/lookup_impl/lookup_table_op_cpu*"]),
    cuda_deps = if_cuda(["//tensorflow_recommenders_addons/dynamic_embedding/core/lib/nvhash:nvhashtable"]),
    cuda_srcs = if_cuda([
        "kernels/cuckoo_hashtable_op_gpu.h",
        "kernels/cuckoo_hashtable_op_gpu.cu.cc",
    ] + glob(["kernels/lookup_impl/lookup_table_op_gpu*"])),
    deps = ["//tensorflow_recommenders_addons/dynamic_embedding/core/lib/cuckoo:cuckoohash"],
)
