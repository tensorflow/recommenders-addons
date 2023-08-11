load(
    "@local_config_tf//:build_defs.bzl",
    "DTF_VERSION_INTEGER",
    "D_GLIBCXX_USE_CXX11_ABI",
    "FOR_TF_SERVING",
    "TF_CXX_STANDARD",
)
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "cuda_is_configured",
    "if_cuda",
    "if_cuda_is_configured",
)

def custom_cuda_op_library(
        name,
        srcs = [],
        cuda_srcs = [],
        deps = [],
        cuda_deps = [],
        copts = [],
        **kwargs):
    if cuda_is_configured():
        custom_op_library(name, srcs, cuda_srcs, deps, cuda_deps, copts, **kwargs)

def custom_op_library(
        name,
        srcs = [],
        cuda_srcs = [],
        deps = [],
        cuda_deps = [],
        copts = [],
        **kwargs):
    if FOR_TF_SERVING == "1":
        deps = deps + [
            "@local_config_tf//:tf_header_lib",
        ]
    else:
        deps = deps + [
            "@local_config_tf//:libtensorflow_framework",
            "@local_config_tf//:tf_header_lib",
        ]

    copts = copts + select({
        "//tensorflow_recommenders_addons:windows": [
            "/DEIGEN_STRONG_INLINE=inline",
            "-DTENSORFLOW_MONOLITHIC_BUILD",
            "/D_USE_MATH_DEFINES",
            "/DPLATFORM_WINDOWS",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
            "/DEIGEN_AVOID_STL_ARRAY",
            "/Iexternal/gemmlowp",
            "/wd4018",
            "/wd4577",
            "/DNOGDI",
            "/UTF_COMPILE_LIBRARY",
        ],
        "//conditions:default": [
            "-pthread",
            "-funroll-loops",
            D_GLIBCXX_USE_CXX11_ABI,
            DTF_VERSION_INTEGER,
        ],
    })
    copts = copts + ["-std=" + TF_CXX_STANDARD]

    if cuda_srcs:
        copts = copts + if_cuda(["-DGOOGLE_CUDA=1"])
        cuda_copts = copts + if_cuda_is_configured([
            "-x cuda",
            "-nvcc_options=relaxed-constexpr",
            "-nvcc_options=ftz=true",
        ])
        cuda_deps = deps + if_cuda_is_configured(cuda_deps) + if_cuda_is_configured([
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart_static",
        ])
        basename = name.split(".")[0]
        native.cc_library(
            name = basename + "_gpu",
            srcs = cuda_srcs,
            deps = cuda_deps,
            copts = cuda_copts,
            alwayslink = 1,
            **kwargs
        )
        deps = deps + if_cuda_is_configured([":" + basename + "_gpu"])

    if FOR_TF_SERVING == "1":
        native.cc_library(
            name = name,
            srcs = srcs,
            copts = copts,
            alwayslink = 1,
            features = select({
                "//tensorflow_recommenders_addons:windows": ["windows_export_all_symbols"],
                "//conditions:default": [],
            }),
            deps = deps,
            **kwargs
        )
    else:
        native.cc_binary(
            name = name,
            srcs = srcs,
            copts = copts,
            linkshared = 1,
            features = select({
                "//tensorflow_recommenders_addons:windows": ["windows_export_all_symbols"],
                "//conditions:default": [],
            }),
            deps = deps,
            **kwargs
        )

def if_cuda_for_tf_serving(if_true, if_false = [], for_tf_serving = "0"):
    """Shorthand for select()'ing on whether we're building with CUDA.

    Returns a select statement which evaluates to if_true if we're building
    with CUDA enabled.  Otherwise, the select statement evaluates to if_false.

    """
    if for_tf_serving == "1":
        return if_true

    return select({
        "@local_config_cuda//cuda:using_nvcc": if_true,
        "@local_config_cuda//cuda:using_clang": if_true,
        "//conditions:default": if_false,
    })
