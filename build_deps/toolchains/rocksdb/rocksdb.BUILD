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

# Enable this to compile RocksDB from source instead.
#make(
#    name = "rocksdb",
#    args = [
#        "EXTRA_CXXFLAGS=\"-fPIC -D_GLIBCXX_USE_CXX11_ABI=0\"",
#        "-j6",
#    ],
#    targets = ["static_lib", "install-static"],
#    lib_source = "@rocksdb//:all_srcs",
#    out_static_libs = ["librocksdb.a"],
#)

# Enable this to use the precompiled library in our image.
cc_library(
    name = "rocksdb",
    includes = ["./include"],
    hdrs = glob(["rocksdb/*.h"]),
    visibility = ["//visibility:public"],
)
cc_import(
    name = "rocksdb_precompiled",
    static_library = "librocksdb.a",
    visibility = ["//visibility:public"],
)
