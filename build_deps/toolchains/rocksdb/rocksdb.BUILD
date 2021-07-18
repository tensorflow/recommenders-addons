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
    make_commands = [
        "make -j`nproc` EXTRA_CXXFLAGS=-fPIC static_lib",
        # TODO: Temporary hack. RocksDB people to fix symlink resolution on their side.
        "cat Makefile | sed 's/\$(FIND) \"include\/rocksdb\" -type f/$(FIND) -L \"include\/rocksdb\" -type f/g' | make -f - static_lib install-static PREFIX=$$INSTALLDIR$$",
    ],
    name = "rocksdb",
    lib_source = "@rocksdb//:all_srcs",
    lib_name = "librocksdb",
)
