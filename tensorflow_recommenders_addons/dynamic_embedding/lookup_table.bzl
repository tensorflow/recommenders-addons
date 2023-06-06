load(
    "@local_config_tf//:build_defs.bzl",
    "DTF_VERSION_INTEGER",
    "D_GLIBCXX_USE_CXX11_ABI",
)

def build_lookup_table_library(
        name,
        srcs = [],
        hdrs = [],
        copts = [],
        deps = [],
        **kwargs):

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
            # "-std=c++17",
            "-std=c++14",
            "-fPIC",
            D_GLIBCXX_USE_CXX11_ABI,
            DTF_VERSION_INTEGER,
        ],
    })

    deps = deps + [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ]

    native.cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        copts = copts,
        deps = deps,
        alwayslink = 1,
        features = select({
            "//tensorflow_recommenders_addons:windows": ["windows_export_all_symbols"],
            "//conditions:default": [],
        }),
        **kwargs
    )