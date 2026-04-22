#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "tensorflow_recommenders_addons/dynamic_embedding/core/kernels/hkv_hashtable_op_gpu.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/cxx17_absl_hack.h"

namespace tensorflow {
namespace recommenders_addons {
namespace hkv_table {

REGISTER_HKV_TABLE(int64, int32);

} // namespace hkv_table
} // namespace recommenders_addons
} // namespace tensorflow
#endif
