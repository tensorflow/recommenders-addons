load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda", "if_cuda_is_configured")

package(default_visibility = ["//visibility:public"])

load("//tensorflow_recommenders_addons:tensorflow_recommenders_addons.bzl", "custom_op_library",
     "if_cuda_for_tf_serving", "tf_nvcc_binary")

cc_library(
    name = "merlin_kv",
    hdrs = [
        "merlin/array_kernels.cuh",
        "merlin/core_kernels.cuh",
        "merlin/debug.hpp",
        "merlin/flexible_buffer.cuh",
        "merlin/memory_pool.cuh",
        "merlin/group_lock.hpp",
        #"merlin/initializers.cuh",
        #"merlin/managed.cuh",
        #"merlin/optimizers.cuh",
        "merlin/types.cuh",
        "merlin/utils.cuh",
        "merlin_hashtable.cuh",
        "merlin_localfile.hpp",
    ],
    deps = [
        "//tensorflow_recommenders_addons/dynamic_embedding/core/lib/utils:cuda_utils",
    ],
)

tf_nvcc_binary(
    name="merlin_kv_insert_and_evict_test",
    srcs=[],
    cuda_srcs=[
        "insert_and_evict_test.cu.cc",
        "test_util.cu.h",
    ],
    cuda_deps=[
        ":merlin_kv",
    ],
    copts=[],
    linkopts=[],
)
