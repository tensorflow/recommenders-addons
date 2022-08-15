load("@local_config_tf//:build_defs.bzl", "DTF_VERSION_INTEGER", "D_GLIBCXX_USE_CXX11_ABI")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "hadoop_file_system",
    srcs = ["hadoop_file_system.cc"],
    hdrs = ["hadoop_file_system.h"],
    copts = [
        D_GLIBCXX_USE_CXX11_ABI,
        DTF_VERSION_INTEGER,
    ],
    deps = [
        "@hadoop",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)
