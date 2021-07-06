load("@rules_foreign_cc//tools/build_defs:make.bzl", "make")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

make(
    name = "hiredis",
    lib_source = "@hiredis//:all_srcs",
    static_libraries = ["libhiredis.a"],
)
