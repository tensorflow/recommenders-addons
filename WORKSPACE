workspace(name = "tf_recommenders_addons")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
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
    sha256 = "3221e1df7bbe95669aecaec1c6c6f4c9a52f472dd4e694e681da951bc738d6bd",
    strip_prefix = "redis-plus-plus-1.2.3",
    url = "https://github.com/sewenew/redis-plus-plus/archive/refs/tags/1.2.3.zip",
)

http_archive(
    name = "hkv",
    build_file = "//build_deps/toolchains/hkv:hkv.BUILD",
    # TODO(LinGeLin) remove this when update hkv
    patch_cmds = [
        """sed -i.bak '1772i\\'$'\\n    ThrustAllocator<uint8_t> thrust_allocator_;\\n' include/merlin_hashtable.cuh""",
        """sed -i.bak '225i\\'$'\\n    thrust_allocator_.set_allocator(allocator_);\\n' include/merlin_hashtable.cuh""",
        "sed -i.bak 's/thrust::sort_by_key(thrust_par.on(stream)/thrust::sort_by_key(thrust_par(thrust_allocator_).on(stream)/' include/merlin_hashtable.cuh",
        "sed -i.bak 's/reduce(thrust_par.on(stream)/reduce(thrust_par(thrust_allocator_).on(stream)/' include/merlin_hashtable.cuh",
        """sed -i.bak '125i\\'$'\\n    template <typename T>\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '126i\\'$'\\n    struct ThrustAllocator : thrust::device_malloc_allocator<T> {\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '127i\\'$'\\n     public:\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '128i\\'$'\\n      typedef thrust::device_malloc_allocator<T> super_t;\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '129i\\'$'\\n      typedef typename super_t::pointer pointer;\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '130i\\'$'\\n      typedef typename super_t::size_type size_type;\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '131i\\'$'\\n     public:\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '132i\\'$'\\n      pointer allocate(size_type n) {\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '133i\\'$'\\n        void* ptr = nullptr;\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '134i\\'$'\\n        MERLIN_CHECK(\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '135i\\'$'\\n            allocator_ != nullptr,\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '136i\\'$'\\n            "[ThrustAllocator] set_allocator should be called in advance!");\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '137i\\'$'\\n        allocator_->alloc(MemoryType::Device, &ptr, sizeof(T) * n);\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '138i\\'$'\\n        return pointer(reinterpret_cast<T*>(ptr));\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '139i\\'$'\\n      }\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '140i\\'$'\\n      void deallocate(pointer p, size_type n) {\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '141i\\'$'\\n        MERLIN_CHECK(\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '142i\\'$'\\n            allocator_ != nullptr,\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '143i\\'$'\\n            "[ThrustAllocator] set_allocator should be called in advance!");\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '144i\\'$'\\n        allocator_->free(MemoryType::Device, reinterpret_cast<void*>(p.get()));\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '145i\\'$'\\n      }\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '146i\\'$'\\n      void set_allocator(BaseAllocator* allocator) { allocator_ = allocator; }\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '147i\\'$'\\n     public:\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '148i\\'$'\\n      BaseAllocator* allocator_ = nullptr;\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '149i\\'$'\\n     };\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '20i\\'$'\\n     #include <thrust/device_malloc_allocator.h>\\n' include/merlin/allocator.cuh""",
        """sed -i.bak '367i\\'$'\\n  for (auto addr : (*table)->buckets_address) {\\n' include/merlin/core_kernels.cuh""",
        """sed -i.bak '368i\\'$'\\n    allocator->free(MemoryType::Device, addr);\\n' include/merlin/core_kernels.cuh""",
        """sed -i.bak '369i\\'$'\\n  }\\n' include/merlin/core_kernels.cuh""",
        """sed -i.bak '370i\\'$'\\n  /*\\n' include/merlin/core_kernels.cuh""",
        """sed -i.bak '382i\\'$'\\n  */\\n' include/merlin/core_kernels.cuh""",
        """sed -i.bak '224i\\'$'\\n  uint8_t* address = nullptr;\\n' include/merlin/core_kernels.cuh""",
        """sed -i.bak '225i\\'$'\\n  allocator->alloc(MemoryType::Device, (void**)&(address), bucket_memory_size * (end - start));\\n' include/merlin/core_kernels.cuh""",
        """sed -i.bak '226i\\'$'\\n  (*table)->buckets_address.push_back(address);\\n' include/merlin/core_kernels.cuh""",
        """sed -i.bak '228i\\'$'\\n    allocate_bucket_others<K, V, S><<<1, 1>>>((*table)->buckets, i, address + (bucket_memory_size * (i-start)), reserve_size, bucket_max_size);\\n' include/merlin/core_kernels.cuh""",
        """sed -i.bak '229i\\'$'\\n    /*\\n' include/merlin/core_kernels.cuh""",
        """sed -i.bak '235i\\'$'\\n    */\\n' include/merlin/core_kernels.cuh""",
        """sed -i.bak '22i\\'$'\\n#include <vector>\\n' include/merlin/types.cuh""",
        """sed -i.bak '143i\\'$'\\n  std::vector<uint8_t*> buckets_address;\\n' include/merlin/types.cuh""",
    ],
    sha256 = "f8179c445a06a558262946cda4d8ae7252d313e73f792586be9b1bc0c993b1cf",
    strip_prefix = "HierarchicalKV-0.1.0-beta.6",
    url = "https://github.com/NVIDIA-Merlin/HierarchicalKV/archive/refs/tags/v0.1.0-beta.6.tar.gz",
)

tf_configure(
    name = "local_config_tf",
)

cuda_configure(name = "local_config_cuda")
