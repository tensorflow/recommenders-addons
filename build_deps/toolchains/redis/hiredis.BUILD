load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")
load(
    "@local_config_tf//:build_defs.bzl",
    "D_GLIBCXX_USE_CXX11_ABI",
)

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cmake_external(
    name = "hiredis",
    cmake_options = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_FLAGS="+D_GLIBCXX_USE_CXX11_ABI,
    ],
    lib_source = "@hiredis//:all_srcs",
    static_libraries = ["libhiredis_static.a"],
)

