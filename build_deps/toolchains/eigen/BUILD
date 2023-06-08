
EIGEN3_THIRD_PARTY_HEADERS = glob([
    "Eigen/Core",
    "Eigen/LU",
    "Eigen/Cholesky",
    "Eigen/Eigenvalues",
    "Eigen/OrderingMethods",
    "Eigen/QR",
    "Eigen/SparseCholesky",
    "Eigen/SparseCore",
    "Eigen/SVD",
    "unsupported/Eigen/MatrixFunctions",
    "unsupported/Eigen/SpecialFunctions",
    "unsupported/Eigen/CXX11/ThreadPool",
    "unsupported/Eigen/CXX11/Tensor",
    "unsupported/Eigen/CXX11/FixedPoint",
]) + glob(["unsupported/Eigen/CXX11/src/FixedPoint/*.h"])

cc_library(
    name = "eigen",
    hdrs = EIGEN3_THIRD_PARTY_HEADERS,
    # includes = if_mkl(["./mkl_include"]),
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen_archive//:eigen3",
    ],
)