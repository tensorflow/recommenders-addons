workspace(name = "tf_recommenders_addons")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//build_deps/tf_dependency:tf_configure.bzl", "tf_configure")
load("//build_deps/toolchains/gpu:cuda_configure.bzl", "cuda_configure")

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

tf_configure(
    name = "local_config_tf",
)

cuda_configure(name = "local_config_cuda")
