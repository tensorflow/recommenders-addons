licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow_recommenders_addons:tensorflow_recommenders_addons.bzl", "custom_op_library")

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
    name = "_redis_table_ops.so",
    srcs = [
        "kernels/redis/redis_impl/redis_connection_pool.hpp",
        "kernels/redis/redis_table_op.h",
        "kernels/redis/redis_table_op.cc",
        "ops/redis_table_ops.cc",
    ]   ,
    deps = [
        "@redis//:HIREDIS",
        "@redis//:REDIS++",
    ]   ,
    copts = ["-pthread", "-O3", "-mavx", "-mavx2", "-mavx512f", "-mfma"],
)