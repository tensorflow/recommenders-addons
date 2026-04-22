workspace(name = "tf_recommenders_addons")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//build_deps/tf_dependency:tf_configure.bzl", "tf_configure")
load("//build_deps/toolchains/gpu:cuda_configure.bzl", "cuda_configure")

http_archive(
    name = "rules_foreign_cc",
    sha256 = "69023642d5781c68911beda769f91fcbc8ca48711db935a75da7f6536b65047f",
    strip_prefix = "rules_foreign_cc-0.6.0",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.6.0.tar.gz",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

# This sets up some common toolchains for building targets. For more details, please see
# https://bazelbuild.github.io/rules_foreign_cc/0.6.0/flatten.html#rules_foreign_cc_dependencies
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
    name = "platforms",
    sha256 = "079945598e4b6cc075846f7fd6a9d0857c33a7afc0de868c2ccb96405225135d",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.4/platforms-0.0.4.tar.gz",
        "https://github.com/bazelbuild/platforms/releases/download/0.0.4/platforms-0.0.4.tar.gz",
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
    build_file = "//build_deps/toolchains/redis:hiredis.BUILD",
    remote = "https://github.com/redis/hiredis.git",
    tag = "v1.1.0",
)

http_archive(
    name = "redis-plus-plus",
    build_file = "//build_deps/toolchains/redis:redis-plus-plus.BUILD",
    sha256 = "630f4e31eaa4a4e3a7f90bd4245022c442bd14b71c8de2623e6a9d7266df44cc",
    strip_prefix = "redis-plus-plus-1.3.15",
    url = "https://github.com/sewenew/redis-plus-plus/archive/refs/tags/1.3.15.zip",
)

http_archive(
    name = "hkv",
    build_file = "//build_deps/toolchains/hkv:hkv.BUILD",
    patch_cmds = [
        "sed -i 's/^__global__ void /static __global__ void /g' include/merlin/core_kernels/group_lock_kernels.cuh",
        "sed -i 's/^uint64_t EMPTY_KEY_CPU/inline uint64_t EMPTY_KEY_CPU/g' include/merlin/types.cuh",
        "sed -i 's/^cudaError_t init_reserved_keys/inline cudaError_t init_reserved_keys/g' include/merlin/types.cuh",
        "sed -i 's/^__global__ void memset64bitKernel/static __global__ void memset64bitKernel/g' include/merlin/utils.cuh",
    ],
    sha256 = "a73d7bea159173db2038f7c5215a7d1fbd5362adfb232fabde206dc64a1e817c",
    strip_prefix = "HierarchicalKV-0.1.0-beta.12",
    url = "https://github.com/NVIDIA-Merlin/HierarchicalKV/archive/refs/tags/v0.1.0-beta.12.tar.gz",
)

http_archive(
    name = "parlayhash",
    build_file_content = """
cc_library(
    name = "parlay_hash",
    hdrs = glob([
        "include/parlay_hash/**/*.h",
        "include/parlay/**/*.h",
        "include/parlay/internal/**/*.h",
        "include/utils/**/*.h",
    ]),
    includes = ["include"],
    copts = ["-mcx16", "-std=c++17"],
    visibility = ["//visibility:public"],
)
""",
    patch_cmds = [
        # Patch 1: sequence_base.h — fix std::addressof(data) ambiguity with union member name
        """sed -i 's/std::byte data\\[1\\]/std::byte storage_data[1]/' include/parlay/internal/sequence_base.h""",
        """sed -i 's/std::addressof(data)/std::addressof(storage_data)/g' include/parlay/internal/sequence_base.h""",
        """sed -i 's/offsetof(header, data)/offsetof(header, storage_data)/g' include/parlay/internal/sequence_base.h""",
        # Patch 2: thread_specific.h — fix non-copyable T in std::function<T(id)> by using placement-new pattern
        """sed -i 's|mutable std::function<T(thread_id_type)> constructor;|mutable std::function<void(void*, thread_id_type)> constructor;|' include/utils/threads/thread_specific.h""",
        """sed -i 's|: constructor(\\[](std::size_t) { return T{}; })|: constructor([](void* p, std::size_t id) { new (p) T{}; })|' include/utils/threads/thread_specific.h""",
        """sed -i 's|: constructor(\\[f = std::forward<F>(constructor_)](std::size_t) { return f(); })|: constructor([f = std::forward<F>(constructor_)](void* p, std::size_t) { new (p) T(f()); })|' include/utils/threads/thread_specific.h""",
        """sed -i 's|explicit ThreadSpecific(F\\&\\& constructor_) : constructor(std::forward<F>(constructor_))|explicit ThreadSpecific(F\\&\\& constructor_) : constructor([f = std::forward<F>(constructor_)](void* p, thread_id_type id) { new (p) T(f(id)); })|' include/utils/threads/thread_specific.h""",
        """sed -i 's|new (static_cast<void\\*>(chunk\\[i\\].get())) T(constructor(chunk_size + i));|constructor(static_cast<void*>(chunk[i].get()), chunk_size + i);|' include/utils/threads/thread_specific.h""",
        """sed -i 's|new (static_cast<void\\*>(chunk\\[i\\].get())) T(constructor(i));|constructor(static_cast<void*>(chunk[i].get()), i);|' include/utils/threads/thread_specific.h""",
    ],
    strip_prefix = "parlayhash-081408ba6111ce78b5add9129f8d902fa2fea58f",
    urls = [
        "https://github.com/cmuparlay/parlayhash/archive/081408ba6111ce78b5add9129f8d902fa2fea58f.tar.gz",
    ],
)

tf_configure(
    name = "local_config_tf",
)

cuda_configure(name = "local_config_cuda")
