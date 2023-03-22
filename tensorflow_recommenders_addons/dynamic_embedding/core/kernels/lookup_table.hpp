#ifndef TFRA_CORE_KERNELS_LOOKUP_TABLE_HPP_
#define TFRA_CORE_KERNELS_LOOKUP_TABLE_HPP_

#include "lookup_table_interface.h"
#include "lookup_table_op_registry.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace recommenders_addons {
namespace lookup_table {

const std::string K_DEFAULT_TABLE_NAME = "STAND_REDIS";

template<class K, class V>
class LookupTableOfTensors final : public LookupTableInterface {
    private:
        mutex mu_;
        LookupTableInterface *lookup_table_ TF_GUARDED_BY(mu_) = nullptr;
    public:
        LookupTableOfTensors(OpKernelContext *ctx, OpKernel *kernel) {
            const std::string device_name = kernel->requested_device();
            std::string table_name;
            TF_CHECK_OK(ReadStringFromEnvVar("LOOKUP_TABLE_NAME", K_DEFAULT_TABLE_NAME, &table_name));
            std::string key_string = tensorflow::DataTypeString(DataTypeToEnum<K>::v());
            std::string value_string = tensorflow::DataTypeString(DataTypeToEnum<V>::v());
            std::string key = table_name + "_" + key_string + "_" + value_string;
            lookup_table_ = op_registry::LookupTableOpRegistry::Global()->LookUp(key, ctx, kernel);
            auto check_table = [](LookupTableInterface *table, const std::string& key){
            return table? Status::OK() : errors::Internal("Failed to lookup table:", key);
            };
            OP_REQUIRES_OK(ctx, check_table(lookup_table_, key));
        }

        ~LookupTableOfTensors() {
            if (lookup_table_) {
            delete lookup_table_;
            }
        }

        size_t size() const {
            return lookup_table_->size();
        }

        Status Find(OpKernelContext *ctx, const Tensor &keys, Tensor *values,
                    const Tensor &default_value)  {
            return lookup_table_->Find(ctx, keys, values, default_value);
        }

        Status FindWithExists(OpKernelContext *ctx, const Tensor &keys,
                                Tensor *values, const Tensor &default_value,
                                Tensor &exists) {
            return lookup_table_->FindWithExists(ctx, keys, values, default_value, exists);
        }

        Status Insert(OpKernelContext *ctx, const Tensor &keys,
                        const Tensor &values) {
            return lookup_table_->Insert(ctx, keys, values);
        }

        Status Accum(OpKernelContext *ctx, const Tensor &keys,
                    const Tensor &values_or_delta, const Tensor &exists) {
            return lookup_table_->Accum(ctx, keys, values_or_delta, exists);
        }

        Status Remove(OpKernelContext *ctx, const Tensor &keys) {
            return lookup_table_->Remove(ctx, keys);
        }

        Status Clear(OpKernelContext *ctx) {
            return lookup_table_->Clear(ctx);
        }

        Status ImportValues(OpKernelContext *ctx, const Tensor &keys,
                            const Tensor &values) {
            return lookup_table_->ImportValues(ctx, keys, values);
        }

        Status ExportValues(OpKernelContext *ctx) {
            return lookup_table_->ExportValues(ctx);
        }

        Status SaveToFileSystem(OpKernelContext *ctx, const string &dirpath,
                                const string &file_name, const size_t buffer_size,
                                bool append_to_file) {
            return lookup_table_->SaveToFileSystem(ctx, dirpath, file_name, buffer_size, append_to_file);
        }

        Status LoadFromFileSystem(OpKernelContext *ctx, const string &dirpath,
                                    const string &file_name, const size_t buffer_size,
                                    bool load_entire_dir) {
            return lookup_table_->LoadFromFileSystem(ctx, dirpath, file_name, buffer_size, load_entire_dir);
        }

        DataType key_dtype() const { 
            return lookup_table_->key_dtype(); }

        DataType value_dtype() const { 
            return lookup_table_->value_dtype(); }

        TensorShape key_shape() const { 
            return lookup_table_->key_shape(); }

        TensorShape value_shape() const { 
            return lookup_table_->value_shape(); }

        tensorflow::int64 MemoryUsed() const {
            return lookup_table_->MemoryUsed();
        }

};

}   // namespace lookup_table
}   // namespace recommenders_addons
}   // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_LOOKUP_TABLE_HPP_