load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")
load(
    "@local_config_tf//:build_defs.bzl",
    "D_GLIBCXX_USE_CXX11_ABI",
    "TF_CXX_STANDARD",
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
    name = "redis-plus-plus",
    cache_entries = {
        # CMake's find_package wants to find cmake config for liba,
        # which we do not have -> disable search
        # "CMAKE_DISABLE_FIND_PACKAGE_HIREDIS": "True",
        # as currently we copy all libraries, built with Bazel, into $EXT_BUILD_DEPS/lib
        # and the headers into $EXT_BUILD_DEPS/include
        "HIREDIS_LIB": "$EXT_BUILD_DEPS/hiredis/lib",
        "HIREDIS_HEADER": "$EXT_BUILD_DEPS/hiredis/include",
    },
    generate_args = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DREDIS_PLUS_PLUS_BUILD_TEST=OFF",
        "-DREDIS_PLUS_PLUS_CXX_STANDARD=" + TF_CXX_STANDARD.split("c++")[-1],
        "-DCMAKE_CXX_FLAGS=" + D_GLIBCXX_USE_CXX11_ABI,
    ],
    lib_source = "@redis-plus-plus//:all_srcs",
    out_static_libs = ["libredis++.a"],
    deps = ["@hiredis"],
)
