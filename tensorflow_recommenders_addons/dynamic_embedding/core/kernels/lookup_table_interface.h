#ifndef TFRA_CORE_KERNELS_LOOKUP_TABLE_INTERFACE_H_
#define TFRA_CORE_KERNELS_LOOKUP_TABLE_INTERFACE_H_

#include "tensorflow/core/framework/lookup_interface.h"

#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/types.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/utils.h"

namespace tensorflow{
namespace recommenders_addons {
namespace lookup_table {

using namespace tensorflow::lookup;

class LookupTableInterface : public LookupInterface {
 public:
   virtual ~LookupTableInterface() = default;
   virtual Status FindWithExists(OpKernelContext *ctx, const Tensor& key,
                     Tensor *values, const Tensor &default_value,
                     Tensor &exists) = 0;
   virtual Status Accum(OpKernelContext *ctx, const Tensor &keys,
                        const Tensor &value_or_delta, const Tensor &exists) = 0;
   virtual Status Clear(OpKernelContext *ctx) = 0;
   virtual Status SaveToFileSystem(OpKernelContext *ctx, const string &dirpath,
                        const string &file_name, const size_t buffer_size,
                        bool append_to_file) = 0;
   virtual Status LoadFromFileSystem(OpKernelContext *ctx, const string &dirpath,
                           const string &file_name, const size_t buffer_size,
                           bool load_entire_dir) = 0;
};   // class LookupTableInterface

}   // namespace lookup_table
}   // namespace recommenders_addons
}   // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_LOOKUP_TABLE_INTERFACE_H_