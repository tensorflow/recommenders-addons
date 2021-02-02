licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_recommenders_addons:tensorflow_recommenders_addons.bzl", "custom_op_library")

custom_op_library(
    name = "_ev_ops.so",
    srcs = [
        "kernels/embedding_var.h",
        "kernels/ev_op_helpers.h",
        "kernels/ev_ops.cc",
        "ops/ev_ops.cc",
    ],
    deps = [
        "@sparsehash_c11//:dense_hash_map",
    ],
)
