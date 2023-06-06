#ifndef TFRA_CORE_KERNELS_LOOKUP_TABLE_HPP_
#define TFRA_CORE_KERNELS_LOOKUP_TABLE_HPP_

// #include "lookup_table_interface.h"
#include "lookup_table_op_registry.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"


namespace tensorflow {
namespace recommenders_addons {
namespace lookup_table {

namespace tfra_lt = tensorflow::recommenders_addons::lookup_table;

const std::string K_DEFAULT_TABLE_NAME = "STAND_REDIS";


class LookupTableInterface : public tensorflow::lookup::LookupInterface {
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
};  // class LookupTableInterface

Status ToTensorflowStatus(const TFRA_Status& status) {
    if (status.ok()) {
        return Status::OK();
    } else {
        switch(status.code()) {
            case StatusCode::CANCELLED:
                return errors::Cancelled(status.message());
            case StatusCode::UNKNOWN:
                return errors::Unknown(status.message());
            case StatusCode::INVALID_ARGUMENT:
                return errors::InvalidArgument(status.message());
            case StatusCode::DEADLINE_EXCEEDED:
                return errors::DeadlineExceeded(status.message());
            case StatusCode::NOT_FOUND:
                return errors::NotFound(status.message());
            case StatusCode::ALREADY_EXISTS:
                return errors::AlreadyExists(status.message());
            case StatusCode::PERMISSION_DENIED:
                return errors::PermissionDenied(status.message());
            case StatusCode::UNAUTHENTICATED:
                return errors::Unauthenticated(status.message());
            case StatusCode::ABORTED:
                return errors::Aborted(status.message());
            case StatusCode::OUT_OF_RANGE:
                return errors::OutOfRange(status.message());
            case StatusCode::UNIMPLEMENTED:
                return errors::Unimplemented(status.message());
            case StatusCode::INTERNAL:
                return errors::Internal(status.message());
            case StatusCode::UNAVAILABLE:
                return errors::Unavailable(status.message());
            case StatusCode::DATA_LOSS:
                return errors::DataLoss(status.message());
            default:
                return errors::Unknown(std::to_string(status.code()), status.message());
        }
    }
}

TFRA_DataType ToTfraDataType(tensorflow::DataType type) {
     typedef tensorflow::DataType tf_dtype;
    switch(type) {
        case tf_dtype::DT_INVALID  : return TFRA_DataType::DT_INVALID;
        case tf_dtype::DT_FLOAT    : return TFRA_DataType::DT_FLOAT;
        case tf_dtype::DT_DOUBLE   : return TFRA_DataType::DT_DOUBLE;
        case tf_dtype::DT_INT32    : return TFRA_DataType::DT_INT32;
        case tf_dtype::DT_UINT8    : return TFRA_DataType::DT_UINT8;
        case tf_dtype::DT_INT16    : return TFRA_DataType::DT_INT16;
        case tf_dtype::DT_INT8     : return TFRA_DataType::DT_INT8;
        case tf_dtype::DT_STRING   : return TFRA_DataType::DT_STRING;
        case tf_dtype::DT_INT64    : return TFRA_DataType::DT_INT64;
        case tf_dtype::DT_BOOL     : return TFRA_DataType::DT_BOOL;
        case tf_dtype::DT_BFLOAT16 : return TFRA_DataType::DT_BFLOAT16;
        case tf_dtype::DT_HALF     : return TFRA_DataType::DT_HALF;
        case tf_dtype::DT_UINT32   : return TFRA_DataType::DT_UINT32;
        case tf_dtype::DT_UINT64   : return TFRA_DataType::DT_UINT64;
        default: return TFRA_DataType::DT_INVALID;
    }
    return TFRA_DataType::DT_INVALID; 
}

template <typename T>
struct ConvertToTfraType_impl {using type = T; };

template <typename T>
using ConvertToTfraType = typename ConvertToTfraType_impl<T>::type;

// #define MATCH_TF_TYPE_AND_TFRA_TYPE(TYPE)    
//   template<>                                 
//   struct ConvertToTfraType_impl<TYPE> {      
//     using type = TYPE;                       
//   }

// MATCH_TF_TYPE_AND_TFRA_TYPE(float);
// MATCH_TF_TYPE_AND_TFRA_TYPE(double);
// MATCH_TF_TYPE_AND_TFRA_TYPE(int32);
// MATCH_TF_TYPE_AND_TFRA_TYPE(uint32);
// MATCH_TF_TYPE_AND_TFRA_TYPE(uint16);
// MATCH_TF_TYPE_AND_TFRA_TYPE(uint8);
// MATCH_TF_TYPE_AND_TFRA_TYPE(int16);
// MATCH_TF_TYPE_AND_TFRA_TYPE(int8);
// MATCH_TF_TYPE_AND_TFRA_TYPE(bool);

// template <>
// struct ConvertToTfraType_impl<tstring> {
//   using type = std::string_view;
// };

// template <>
// struct MODIFIED_TO_TFRA_TYPE<int64_t> {
//   typedef  Type;
// };


template<class K, class V>
class LookupTableOfTensors final : public LookupTableInterface {
    private:
        mutex mu_;
        // TFRALookupTableInterface<K, V> *lookup_table_ TF_GUARDED_BY(mu_) = nullptr;
        // TFRALookupTableBase *lookup_table_base_ TF_GUARDED_BY(mu_) = nullptr;
        TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>> *lookup_table_ TF_GUARDED_BY(mu_) = nullptr;
        TFRA_DataType key_dtype_;
        TensorShape value_shape_;

        Status ParseKVTableInfo(OpKernelContext *ctx, OpKernel *kernel, KVTableInfo& info) {
            TF_RETURN_IF_ERROR(GetNodeAttr(kernel->def(), "embedding_name", &info.embedding_name_));
            LOG(INFO) << "embedding name: " << info.embedding_name_;
            TensorShape value_shape;
            TF_RETURN_IF_ERROR(GetNodeAttr(kernel->def(), "value_shape", &value_shape_));
            LOG(INFO) << "value_shape dims: " << value_shape_.dims();
            LOG(INFO) << "value shape debug string: " << value_shape_.DebugString();
            LOG(INFO) << "value shape num elements: " << value_shape_.num_elements();
            if (!TensorShapeUtils::IsVector(value_shape_)) {
                errors::InvalidArgument("Default value must be a vector, got shape ",
                                value_shape_.DebugString());
            }
            for (int i = 0; i<value_shape_.dims(); i++) {
                info.value_shape_.push_back(value_shape_.dim_size(i));
            }

            TF_RETURN_IF_ERROR(GetNodeAttr(kernel->def(), "embedding_name", &info.embedding_name_));
            TF_RETURN_IF_ERROR(GetNodeAttr(kernel->def(), "redis_config_abs_dir", &info.redis_config_.redis_config_abs_dir_));
            TF_RETURN_IF_ERROR(GetNodeAttr(kernel->def(), "redis_config_abs_dir_env", &info.redis_config_.redis_config_abs_dir_env_));

            return Status::OK();
        }
    public:
        LookupTableOfTensors(OpKernelContext *ctx, OpKernel *kernel) {
            KVTableInfo info;
            OP_REQUIRES_OK(ctx, ParseKVTableInfo(ctx, kernel, info));

            const std::string device_name = kernel->requested_device();
            std::string table_name;
            TF_CHECK_OK(tensorflow::ReadStringFromEnvVar("LOOKUP_TABLE_NAME", K_DEFAULT_TABLE_NAME, &table_name));
            std::string key_string = tensorflow::DataTypeString(tensorflow::DataTypeToEnum<K>::v());
            std::string value_string = tensorflow::DataTypeString(tensorflow::DataTypeToEnum<V>::v());

            LOG(INFO) << "key string: " << key_string << " value string: " << value_string;

            // key_dtype_ = ToTfraDataType(tensorflow::DataTypeToEnum<K>::v());
            // value_dtype_ = ToTfraDataType(tensorflow::DataTypeToEnum<V>::v());



            // if (key_string == "string") {
            //     if (value_string == "string") {
            //         else 
            //     }

            // } else if (key_string == "bfloat16") {

            // } else if (key_string == "half") {

            // } else {
            //     lookup_table_base_ = op_registry::LookupTableOpRegistry<K, V>::Global()->Lookup(key, info);
            // }

            std::string key = table_name + "_" + key_string + "_" + value_string;

            lookup_table_ = op_registry::LookupTableOpRegistry<ConvertToTfraType<K>, ConvertToTfraType<V>>::Global()->LookUp(key, info);

            // lookup_table_ = op_registry::LookupTableOpRegistry<K, V>::Global()->LookUp(key, info);
            auto check_table = [this](const std::string& key){
                return lookup_table_? Status::OK() : errors::Internal("Failed to lookup table:", key);
            };
            OP_REQUIRES_OK(ctx, check_table(key));
        }

        ~LookupTableOfTensors() {
            if (lookup_table_) {
                delete lookup_table_;
            }
        }

        size_t size() const {
            // auto lookup_table_ = dynamic_cast<TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>>*>(lookup_table_base_);
            return lookup_table_->size();
        }

        Status Find(OpKernelContext *ctx, const Tensor &keys, Tensor *values,
                    const Tensor &default_value)  {
            DeviceNameUtils::ParsedName parsed;
            DeviceNameUtils::ParseFullName(ctx->device()->name(), &parsed);
            std::string device_type = parsed.type;

            if (device_type == "CPU") {
                // const auto key_flat = keys.flat<K>();
                // auto value_flat = values->flat<V>();
                // const auto devault_flat = default_value.flat<V>();
            // auto lookup_table_ = dynamic_cast<TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>>*>(lookup_table_base_);

                if (keys.NumElements() > 0) {
                    return ToTensorflowStatus(lookup_table_->Find((ConvertToTfraType<K>*)keys.tensor_data().data(), keys.NumElements(),
                        (ConvertToTfraType<V>*)values->tensor_data().data(), values->NumElements() / keys.NumElements(), 
                        (ConvertToTfraType<V>*)default_value.tensor_data().data(), default_value.NumElements()));
                } else {
                    // TODO(chenjinglin) Better error prompt
                    return Status::OK();
                    // return errors::InvalidArgument("Got empty keys!");
                }



            } else if (device_type == "GPU") {
                // TODO chenjinglin
                return Status::OK();
            } else {
                return errors::InvalidArgument("Unsupported Device Type: ", device_type);
            }

            // return ToTensorflowStatus(lookup_table_->Find(ctx, keys, values, default_value));


        }

        Status FindWithExists(OpKernelContext *ctx, const Tensor &keys,
                                Tensor *values, const Tensor &default_value,
                                Tensor &exists) {
            // auto lookup_table_ = dynamic_cast<TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>>*>(lookup_table_base_);

            return ToTensorflowStatus(lookup_table_->FindWithExists((ConvertToTfraType<K>*)keys.tensor_data().data(), keys.NumElements(), 
            (ConvertToTfraType<V>*)values->tensor_data().data(), values->NumElements() / keys.NumElements(),
            (ConvertToTfraType<V>*)default_value.tensor_data().data(), default_value.NumElements(), (bool*)exists.tensor_data().data()));
        }

        Status Insert(OpKernelContext *ctx, const Tensor &keys,
                        const Tensor &values) {
            // auto lookup_table_ = dynamic_cast<TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>>*>(lookup_table_base_);

            LOG(INFO) << "keys: " << keys.DebugString();
            LOG(INFO) << "values: " << values.DebugString();

            return ToTensorflowStatus(lookup_table_->Insert((ConvertToTfraType<K>*)keys.tensor_data().data(), keys.NumElements(),
            (ConvertToTfraType<V>*)values.tensor_data().data(), values.NumElements() / keys.NumElements()));
        }

        Status Accum(OpKernelContext *ctx, const Tensor &keys,
                    const Tensor &values_or_delta, const Tensor &exists) {
            // auto lookup_table_ = dynamic_cast<TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>>*>(lookup_table_base_);

            LOG(INFO) << "accum keys: " << keys.DebugString();
            LOG(INFO) << "value or delta: " << values_or_delta.DebugString();
            
            return ToTensorflowStatus(lookup_table_->Accum((ConvertToTfraType<K>*)keys.tensor_data().data(), keys.NumElements(),
             (ConvertToTfraType<V>*)values_or_delta.tensor_data().data(), values_or_delta.NumElements(), (bool*)exists.tensor_data().data()));
        }

        Status Remove(OpKernelContext *ctx, const Tensor &keys) {
            // auto lookup_table_ = dynamic_cast<TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>>*>(lookup_table_base_);

            return ToTensorflowStatus(lookup_table_->Remove((ConvertToTfraType<K>*)keys.tensor_data().data(), keys.NumElements()));
        }

        Status Clear(OpKernelContext *ctx) {
            // auto lookup_table_ = dynamic_cast<TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>>*>(lookup_table_base_);

            return ToTensorflowStatus(lookup_table_->Clear());
        }

        Status ImportValues(OpKernelContext *ctx, const Tensor &keys,
                            const Tensor &values) {
            // auto lookup_table_ = dynamic_cast<TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>>*>(lookup_table_base_);

            return ToTensorflowStatus(lookup_table_->ImportValues((ConvertToTfraType<K>*)keys.tensor_data().data(), keys.NumElements(), 
            (ConvertToTfraType<V>*)values.tensor_data().data(), values.NumElements() / keys.NumElements()));
        }

        Status ExportValues(OpKernelContext *ctx) {
            // auto lookup_table_ = dynamic_cast<TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>>*>(lookup_table_base_);

            // TODO chenjinglin TensorShape 的total size 需要int64_t
            int64_t total_size = lookup_table_->size();
            auto dim = lookup_table_->dim();
            LOG(INFO) << "Export dim: " << lookup_table_->dim();
            Tensor *keys;
            TF_RETURN_IF_ERROR(
                ctx->allocate_output("keys", TensorShape({total_size}), &keys));

            Tensor *values;
            TF_RETURN_IF_ERROR(ctx->allocate_output(
                "values", TensorShape({total_size, dim}), &values));
            return ToTensorflowStatus(lookup_table_->ExportValues((ConvertToTfraType<K>*)keys->tensor_data().data(),
                                     (ConvertToTfraType<V>*)values->tensor_data().data()));
        }

        Status SaveToFileSystem(OpKernelContext *ctx, const string &dirpath,
                                const string &file_name, const size_t buffer_size,
                                bool append_to_file) {
            // auto lookup_table_ = dynamic_cast<TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>>*>(lookup_table_base_);

            // get filesystem
            string filepath = io::JoinPath(dirpath, file_name);
            FileSystem *fs;
            const auto env = ctx->env();
            TF_RETURN_WITH_CONTEXT_IF_ERROR(
                env->GetFileSystemForFile(filepath, &fs),
                "Please make sure you have already imported tensorflow_io before using "
                "TFRA file system operation.");

            size_t total_size = lookup_table_->size();

            // long long cursor = 0;
            // std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> hscan_reply;
            // const redisReply *kvs_reply;


            // construct file system relative object
            std::unique_ptr<WritableFile> key_writer;
            std::unique_ptr<WritableFile> value_writer;
            const string key_filepath(filepath + "-keys");
            const string value_filepath(filepath + "-values");
            string key_tmpfilepath(filepath + "-keys.tmp");
            string value_tmpfilepath(filepath + "-values.tmp");

            bool has_atomic_move = false;
            auto has_atomic_move_ret = fs->HasAtomicMove(filepath, &has_atomic_move);
            bool need_tmp_file =
                (has_atomic_move == false) || (has_atomic_move_ret != Status::OK());
            if (!need_tmp_file) {
            key_tmpfilepath = key_filepath;
            value_tmpfilepath = value_filepath;
            }

            TF_RETURN_IF_ERROR(fs->RecursivelyCreateDir(std::string(fs->Dirname(filepath))));
            if (append_to_file) {
                TF_RETURN_IF_ERROR(fs->NewAppendableFile(key_tmpfilepath, &key_writer));
                TF_RETURN_IF_ERROR(fs->NewAppendableFile(value_tmpfilepath, &value_writer));
            } else {
                TF_RETURN_IF_ERROR(fs->NewWritableFile(key_tmpfilepath, &key_writer));
                TF_RETURN_IF_ERROR(fs->NewWritableFile(value_tmpfilepath, &value_writer));
            }

            if (total_size == 0) {
                // TODO chenjinglin
                // LOG(WARNING) << "There is no embedding table called " << keys_prefix_name
                //             << " existing in the Redis service. "
                //             << "Saving values to file system failed.";
                LOG(WARNING) << "There is no embedding table called "
                            << " existing in the Redis service. "
                            << "Saving values to file system failed.";
                return Status::OK();
            }

            // buffer for write to file system
            std::vector<int> shape_vec = lookup_table_->value_shape();
            int value_dim = 0;
            for(auto dim : shape_vec) {
                value_dim += dim;
            }
            const size_t value_len = sizeof(V) * value_dim;
            const size_t key_buffer_byte_size = buffer_size * sizeof(K);
            const size_t value_buffer_byte_size = buffer_size * value_len;
            
            std::vector<char> key_buffer_vector(key_buffer_byte_size);
            std::vector<char> value_buffer_vector(value_buffer_byte_size);
            
            // const K *key_buffer = reinterpret_cast<const K *>(key_buffer_vector.data());
            // const V *value_buffer = reinterpret_cast<const V *>(value_buffer_vector.data());

            K *key_buffer = (K*)key_buffer_vector.data();
            V *value_buffer = (V*)value_buffer_vector.data();

            size_t total_saved = 0;
            size_t search_offset = 0;
            while(total_saved < total_size) {
                size_t dumped_counter = 0;
                TF_RETURN_IF_ERROR(
                    ToTensorflowStatus(
                        lookup_table_->Dump((ConvertToTfraType<K>*)key_buffer, (ConvertToTfraType<V>*)value_buffer, search_offset, buffer_size, &dumped_counter)));
                
                if (dumped_counter > 0) {
                    search_offset += dumped_counter;
                    total_saved += dumped_counter;
                    size_t key_offset = dumped_counter * sizeof(K);
                    size_t value_offset = dumped_counter * value_len;
                    
                    TF_RETURN_IF_ERROR(key_writer->Append(StringPiece(key_buffer_vector.data(), key_offset)));
                    TF_RETURN_IF_ERROR(value_writer->Append(StringPiece(value_buffer_vector.data(), value_offset)));
                }
            }

            TF_RETURN_IF_ERROR(key_writer->Flush());
            TF_RETURN_IF_ERROR(value_writer->Flush());
            TF_RETURN_IF_ERROR(key_writer->Sync());
            TF_RETURN_IF_ERROR(value_writer->Sync());

            LOG(INFO) << "Finish saving " << total_saved << " keys and values to "
                    << key_filepath << " and " << value_filepath << " in total.";

            if (need_tmp_file) {
                TF_RETURN_IF_ERROR(fs->FileExists(key_tmpfilepath));
                TF_RETURN_IF_ERROR(fs->RenameFile(key_tmpfilepath, key_filepath));
                TF_RETURN_IF_ERROR(fs->FileExists(value_tmpfilepath));
                TF_RETURN_IF_ERROR(fs->RenameFile(value_tmpfilepath, value_filepath));
            }

            return Status::OK();
        }

        Status LoadFromFileSystemImpl(FileSystem *fs, const string &filepath,
                                      const size_t buffer_size) {
            // auto lookup_table_ = dynamic_cast<TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>>*>(lookup_table_base_);

            const string key_filepath = filepath + "-keys";
            TF_RETURN_IF_ERROR(fs->FileExists(key_filepath));
            std::unique_ptr<RandomAccessFile> key_file;
            TF_RETURN_IF_ERROR(fs->NewRandomAccessFile(key_filepath, &key_file));
            std::unique_ptr<io::RandomAccessInputStream> key_input_stream(
                new io::RandomAccessInputStream(key_file.get()));
            size_t key_buffer_byte_size = buffer_size * sizeof(K);
            io::BufferedInputStream key_reader(key_input_stream.get(),
                                            key_buffer_byte_size * 2);

            const string value_filepath = filepath + "-values";
            TF_RETURN_IF_ERROR(fs->FileExists(key_filepath));
            std::unique_ptr<RandomAccessFile> value_file;
            TF_RETURN_IF_ERROR(fs->NewRandomAccessFile(value_filepath, &value_file));
            std::unique_ptr<io::RandomAccessInputStream> value_input_stream(
                new io::RandomAccessInputStream(value_file.get()));
            

            auto runtime_value_dim = lookup_table_->dim();
            const size_t value_len = sizeof(V) * runtime_value_dim;
            size_t value_buffer_byte_size = buffer_size * value_len;
            io::BufferedInputStream value_reader(value_input_stream.get(),
                                                value_buffer_byte_size * 2);

            uint64 key_file_size = 0;
            TF_RETURN_IF_ERROR(fs->GetFileSize(key_filepath, &key_file_size));
            const size_t key_size = key_file_size / sizeof(K);

            uint64 value_file_size = 0;
            TF_RETURN_IF_ERROR(fs->GetFileSize(value_filepath, &value_file_size));
            const size_t value_size = value_file_size / value_len;

            if (key_size != value_size) {
                return errors::Unavailable(
                    "the keys number in file " + key_filepath +
                    " is not equal to the value vectors number in file " +
                    value_filepath + ".");
            }

            tstring key_buffer;
            key_buffer.resize(key_buffer_byte_size);
            tstring value_buffer;
            value_buffer.resize(value_buffer_byte_size);

            size_t key_file_offset = 0;
            int64_t remainder = key_file_size - key_file_offset;
            size_t nkeys = 0;
            size_t key_read_byte = 0;
            size_t value_read_byte = 0;
            while (remainder > 0) {
                if (remainder > static_cast<int64_t>(key_buffer_byte_size)) {
                    key_read_byte = key_buffer_byte_size;
                    nkeys = buffer_size;
                    value_read_byte = value_buffer_byte_size;
                } else {
                    key_read_byte = remainder;
                    nkeys = key_read_byte / sizeof(K);
                    value_read_byte = nkeys * value_len;
                }
                TF_RETURN_IF_ERROR(key_reader.ReadNBytes(key_read_byte, &key_buffer));
                TF_RETURN_IF_ERROR(value_reader.ReadNBytes(value_read_byte, &value_buffer));
                TF_RETURN_IF_ERROR(ToTensorflowStatus(lookup_table_->Insert((ConvertToTfraType<K> *)key_buffer.data(), nkeys,
                                            (ConvertToTfraType<V> *)value_buffer.data(),
                                            runtime_value_dim)));
                key_file_offset += key_read_byte;
                remainder = key_file_size - key_file_offset;
            }

            LOG(INFO) << "Finish loading " << key_size << " keys and values from "
                    << key_filepath << " and " << value_filepath << " in total.";

            return Status::OK();
        }

        Status LoadFromFileSystem(OpKernelContext *ctx, const string &dirpath,
                                    const string &file_name, const size_t buffer_size,
                                    bool load_entire_dir) {
            // auto lookup_table_ = dynamic_cast<TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>>*>(lookup_table_base_);

            string filepath = io::JoinPath(dirpath, file_name);
            FileSystem *fs;
            const auto env = ctx->env();
            TF_RETURN_WITH_CONTEXT_IF_ERROR(
                env->GetFileSystemForFile(filepath, &fs),
                "Please make sure you have already imported tensorflow_io before using "
                "TFRA file system operation.");

            if (load_entire_dir) {
                string separator = "_mht_";
                int separator_pos = file_name.rfind(separator);
                string file_pattern =
                    io::JoinPath(dirpath,
                                file_name.substr(0, separator_pos + separator.size())) + "*";
                std::vector<string> all_filepath;
                TF_RETURN_IF_ERROR(fs->GetMatchingPaths(file_pattern, &all_filepath));
                // delete -keys/-values postfix
                for (auto it = all_filepath.begin(); it != all_filepath.end(); ++it) {
                    int kv_separator_pos = it->rfind("-");
                    *it = it->substr(0, kv_separator_pos);
                }
                // remove duplicate elements
                sort(all_filepath.begin(), all_filepath.end());
                all_filepath.erase(unique(all_filepath.begin(), all_filepath.end()),
                                    all_filepath.end());
                for (auto &fp : all_filepath) {
                    TF_RETURN_IF_ERROR(LoadFromFileSystemImpl(fs, fp, buffer_size));
                }
            } else {
                return LoadFromFileSystemImpl(fs, filepath, buffer_size);
            }
            return Status::OK();
        }

        DataType key_dtype() const { 
            // auto lookup_table_ = dynamic_cast<TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>>*>(lookup_table_base_);

            return DataType((int)lookup_table_->key_dtype()); }

        DataType value_dtype() const { 
            // auto lookup_table_ = dynamic_cast<TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>>*>(lookup_table_base_);

            return DataType((int)lookup_table_->value_dtype()); }

        TensorShape key_shape() const {
            // auto lookup_table_ = dynamic_cast<TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>>*>(lookup_table_base_);
            return TensorShape();
            // TensorShape shape;
            // std::vector<int> shape_vec = lookup_table_->key_shape(); 
            // for(auto dim : shape_vec) {
            //     shape.AddDim(dim);
            // }
            // return shape;
        }

        TensorShape value_shape() const {
            // auto lookup_table_ = dynamic_cast<TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>>*>(lookup_table_base_);
            return value_shape_;
            // TensorShape shape;
            // std::vector<int> shape_vec = lookup_table_->value_shape(); 
            // LOG(INFO) << "BEFOREADDDIM";
            // for(auto dim : shape_vec) {
            //     LOG(INFO) << "func value_shape dim: " << dim << std::endl;
            //     shape.AddDim(dim);
            // }
            // LOG(INFO) << "DIMS: " << shape.dims();
            // return shape;
        }

        tensorflow::int64 MemoryUsed() const {
            // auto lookup_table_ = dynamic_cast<TFRALookupTableInterface<ConvertToTfraType<K>, ConvertToTfraType<V>>*>(lookup_table_base_);

            return lookup_table_->MemoryUsed();
        }
};

}   // namespace lookup_table_
}   // namespace recommenders_addons
}   // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_LOOKUP_TABLE_HPP_