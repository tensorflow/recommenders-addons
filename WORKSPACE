workspace(name = "tf_recommenders_addons")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("//build_deps/tf_dependency:tf_configure.bzl", "tf_configure")
load("//build_deps/toolchains/gpu:cuda_configure.bzl", "cuda_configure")

http_archive(
    name = "rules_foreign_cc",
    sha256 = "9f96cae32dca0578960468a5a9f66e1ddb8cba8cc210e3e8198f70a20dba45d1",
    strip_prefix = "rules_foreign_cc-0.0.9",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/0.0.9.zip",
)

load("@rules_foreign_cc//:workspace_definitions.bzl", "rules_foreign_cc_dependencies")

# This sets up some common toolchains for building targets. For more details, please see
# https://bazelbuild.github.io/rules_foreign_cc/0.3.0/flatten.html#rules_foreign_cc_dependencies
rules_foreign_cc_dependencies(register_default_tools = True)

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

new_git_repository(
    name = "hiredis",
    branch = "master",
    build_file = "//build_deps/toolchains/redis:hiredis.BUILD",
    remote = "https://github.com/redis/hiredis.git",
)

http_archive(
    name = "redis-plus-plus",
    build_file = "//build_deps/toolchains/redis:redis-plus-plus.BUILD",
    sha256 = "3221e1df7bbe95669aecaec1c6c6f4c9a52f472dd4e694e681da951bc738d6bd",
    strip_prefix = "redis-plus-plus-1.2.3",
    url = "https://github.com/sewenew/redis-plus-plus/archive/refs/tags/1.2.3.zip",
)

tf_configure(
    name = "local_config_tf",
)

cuda_configure(name = "local_config_cuda")
