load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "all_hdrs",
    srcs = glob(["*.h"]),
    visibility = ["//visibility:public"],
)

cmake_external(
    name = "redis-plus-plus",
    cache_entries = {
        # CMake's find_package wants to find cmake config for liba,
        # which we do not have -> disable search
        # "CMAKE_DISABLE_FIND_PACKAGE_HIREDIS": "True",
        # as currently we copy all libraries, built with Bazel, into $EXT_BUILD_DEPS/lib
        # and the headers into $EXT_BUILD_DEPS/include
        "HIREDIS_LIB": "$EXT_BUILD_DEPS",
    },
    cmake_options = ["-DCMAKE_BUILD_TYPE=Release","-DCMAKE_PREFIX_PATH=$EXT_BUILD_DEPS","-DREDIS_PLUS_PLUS_BUILD_TEST=OFF"],
    lib_source = "@redis-plus-plus//:all_srcs",
    out_include_dir = "@redis-plus-plus//:all_hdrs",
    static_libraries = ["libredis++.a"],
    deps = ["@hiredis//:hiredis"]
)
