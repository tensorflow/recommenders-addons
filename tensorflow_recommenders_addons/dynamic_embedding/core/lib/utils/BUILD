package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuda_utils",
    hdrs = [
      "cuda_utils.cuh",
    ],
    linkopts = [
        "-L/usr/local/cuda-11.2/targets/x86_64-linux/lib/stubs",
        "-lnvidia-ml",
    ],
)
