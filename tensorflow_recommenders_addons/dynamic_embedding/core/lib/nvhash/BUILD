package(default_visibility = ["//visibility:public"])

cc_library(
    name = "nvhashtable",
    hdrs = [
        "cudf/concurrent_unordered_map.cuh",
        "cudf/hash_functions.cuh",
        "cudf/managed.cuh",
        "cudf/managed_allocator.cuh",
        "nv_hashtable.cuh",
        "nv_util.h",
    ] + glob(["cub/cub/**/*cuh"]),
)
