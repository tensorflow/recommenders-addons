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
        "CMAKE_DISABLE_FIND_PACKAGE_LIBA": "True",
        # as currently we copy all libraries, built with Bazel, into $EXT_BUILD_DEPS/lib
        # and the headers into $EXT_BUILD_DEPS/include
        "LIBA_DIR": "$EXT_BUILD_DEPS",
    },
    cmake_options = ["-DCMAKE_BUILD_TYPE=Release"],
    lib_source = "@redis-plus-plus//:all_srcs",
    out_include_dir = "@redis-plus-plus//:all_hdrs",
    static_libraries = ["libredis++.a"],
    deps = ["@hiredis//:hiredis"]
)
