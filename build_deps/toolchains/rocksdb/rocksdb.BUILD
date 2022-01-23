load("@rules_foreign_cc//foreign_cc:defs.bzl", "make")

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
    name = "rocksdb",
    args = [
        "EXTRA_CXXFLAGS=\"-fPIC -D_GLIBCXX_USE_CXX11_ABI=0\"",
        "-j6",
    ],
    targets = ["static_lib", "install-static"],
    lib_source = "@rocksdb//:all_srcs",
    out_static_libs = ["librocksdb.a"],
)
