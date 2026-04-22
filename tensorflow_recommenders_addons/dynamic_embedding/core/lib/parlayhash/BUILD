package(default_visibility = ["//visibility:public"])

cc_library(
    name = "concurrent_hashmap",
    hdrs = [
        "concurrent_hash_map.h",
    ],
    copts = [
        "-std=c++17",
        "-mcx16",
    ],
    deps = [
        "@parlayhash//:parlay_hash",
    ],
)
