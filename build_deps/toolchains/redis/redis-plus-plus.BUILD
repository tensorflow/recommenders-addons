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
    cmake_options = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DREDIS_PLUS_PLUS_BUILD_TEST=OFF",
        "-DREDIS_PLUS_PLUS_CXX_STANDARD=11",
        "-DCMAKE_CXX_FLAGS="+D_GLIBCXX_USE_CXX11_ABI,
    ],
    lib_source = "@redis-plus-plus//:all_srcs",
    static_libraries = ["libredis++.a"],
    deps = ["@hiredis"],
)
