workspace(name = "tf_recommenders_addons")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

load("//build_deps/tf_dependency:tf_configure.bzl", "tf_configure")
load("//build_deps/toolchains/gpu:cuda_configure.bzl", "cuda_configure")

git_repository(
    name = "bazel_skylib",
    remote = "git@github.com:bazelbuild/bazel-skylib.git",
    tag = "0.1.0",  # change this to use a different release
)

# Rule repository, note that it's recommended to use a pinned commit to a released version of the rules
http_archive(
   name = "rules_foreign_cc",
   sha256 = "c2cdcf55ffaf49366725639e45dedd449b8c3fe22b54e31625eb80ce3a240f1e",
   strip_prefix = "rules_foreign_cc-0.1.0",
   url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.1.0.zip",
)

http_archive(
    name = "cub_archive",
    build_file = "//build_deps/toolchains/gpu:cub.BUILD",
    sha256 = "6bfa06ab52a650ae7ee6963143a0bbc667d6504822cbd9670369b598f18c58c3",
    strip_prefix = "cub-1.8.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NVlabs/cub/archive/1.8.0.zip",
        "https://github.com/NVlabs/cub/archive/1.8.0.zip",
    ],
)

http_archive(
    name = "sparsehash_c11",
    build_file = "//third_party:sparsehash_c11.BUILD",
    sha256 = "d4a43cad1e27646ff0ef3a8ce3e18540dbcb1fdec6cc1d1cb9b5095a9ca2a755",
    strip_prefix = "sparsehash-c11-2.11.1",
    urls = [
        "https://github.com/sparsehash/sparsehash-c11/archive/v2.11.1.tar.gz",
    ],
)

_ALL_CONTENT = """\
filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
"""

http_archive(
   name = "hiredis",
   build_file_content = _ALL_CONTENT,
   build_file = "//build_deps/toolchains/redis:hiredis.BUILD",
   sha256 = "71fded144c038ce911d7745e22d901daa226e1b8f023e60f87a499356f77befa",
   strip_prefix = "hiredis-1.0.0",
   url = "https://github.com/redis/hiredis/archive/refs/tags/v1.0.0.zip",
)

http_archive(
   name = "redis-plus-plus",
   build_file_content = _ALL_CONTENT,
   build_file = "//build_deps/toolchains/redis:redis-plus-plus.BUILD",
   sha256 = "3221e1df7bbe95669aecaec1c6c6f4c9a52f472dd4e694e681da951bc738d6bd",
   strip_prefix = "redis-plus-plus-1.2.3",
   url = "https://github.com/sewenew/redis-plus-plus/archive/refs/tags/1.2.3.zip",
)

# new_local_repository(
#     name = "redis",
#     path = "/usr",
#     build_file_content = """
# package(default_visibility = ["//visibility:public"])
# load("@bazel_skylib//lib:collections.bzl", "collections")
# cc_import(
#     name = "HIREDIS",
#     hdrs = collections.uniq(glob(["local/include/hiredis/*.h"])+glob(["local/include/hiredis/**/*.h"])) ,
#     static_library = "local/lib/libhiredis.a",
#     shared_library = "local/lib/libhiredis.so",
# )
# cc_import(
#     name = "REDIS++",
#     hdrs = glob(["local/include/sw/redis++/*.h"]) ,
#     static_library = "local/lib/libredis++.a",
#     shared_library = "local/lib/libredis++.so",
# )
# """
# )

tf_configure(
    name = "local_config_tf",
)

cuda_configure(name = "local_config_cuda")
