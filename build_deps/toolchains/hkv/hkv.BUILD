load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda", "if_cuda_is_configured")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "hkv",
    hdrs = glob([
        "include/merlin/core_kernels/*.cuh",
        "include/merlin/*.cuh",
        "include/merlin_hashtable.cuh",
        "include/merlin_localfile.hpp",
    ]),
    copts = [
        "-Ofast",
    ],
    include_prefix = "include",
    includes = ["include"],
)