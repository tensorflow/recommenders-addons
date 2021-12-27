load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")
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

cmake(
    name = "hiredis",
    generate_args = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_FLAGS="+D_GLIBCXX_USE_CXX11_ABI,
    ],
    lib_source = "@hiredis//:all_srcs",
    out_static_libs = ["libhiredis_static.a"],
)