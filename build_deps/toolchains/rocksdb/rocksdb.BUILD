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
        # Uncomment
        # "make -j`nproc` EXTRA_CXXFLAGS=\"-fPIC -D_GLIBCXX_USE_CXX11_ABI=0\" rocksdbjavastatic_deps",
        # to build static dependencies in $$BUILD_TMPDIR$$.
        "make -j`nproc` EXTRA_CXXFLAGS=\"-fPIC -D_GLIBCXX_USE_CXX11_ABI=0\" static_lib",
        # TODO: Temporary hack. RocksDB people to fix symlink resolution on their side.
        "cat Makefile | sed 's/\$(FIND) \"include\/rocksdb\" -type f/$(FIND) -L \"include\/rocksdb\" -type f/g' | make -f - static_lib install-static PREFIX=$$INSTALLDIR$$",
    ],
    name = "rocksdb",
    lib_source = "@rocksdb//:all_srcs",
    lib_name = "librocksdb",
)
