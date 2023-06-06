#ifndef TFRA_CORE_KERNELS_LOOKUP_TABLE_INTERFACE_H_
#define TFRA_CORE_KERNELS_LOOKUP_TABLE_INTERFACE_H_

// #include "tensorflow/core/framework/lookup_interface.h"
#include <vector>

#include "tensorflow_recommenders_addons/dynamic_embedding/core/kernels/lookup_table_types.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/types.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/utils.h"

namespace tensorflow{
namespace recommenders_addons {
namespace lookup_table {

// using namespace tensorflow::lookup;

typedef struct RedisConfig {
  std::string redis_config_abs_dir_;
  std::string redis_config_abs_dir_env_;

} RedisConfig;

typedef struct KVTableInfo {
  std::vector<int> value_shape_;

  std::string embedding_name_;
  RedisConfig redis_config_;
} KVTableInfo;

// class TFRALookupTableBase {
// public:
//   TFRALookupTableBase() {}
//   virtual ~TFRALookupTableBase() = 0;
// };

template<class K, class V>
class TFRALookupTableInterface {
 public:
   virtual ~TFRALookupTableInterface() = default;
   virtual size_t size() const = 0;
   virtual int64_t dim() const = 0;
   virtual TFRA_Status Find(const K* keys, int64_t key_num, V* const values, int64_t value_dim,
                            const V* default_values, int64_t default_value_num) = 0;
   virtual TFRA_Status Insert(const K* keys, int64_t key_num, const V* values, int64_t value_dim) = 0;
   virtual TFRA_Status Remove(const K* keys, int64_t key_num) = 0;
   virtual TFRA_Status ImportValues(const K* keys, int64_t key_num,
                                    const V* values, int64_t value_dim) = 0;
   virtual TFRA_Status ExportValues(K* const keys, V* const values) = 0;
   virtual TFRA_Status FindWithExists(const K* keys, int64_t key_num,
                                      V* const values, int64_t value_dim,
                                      const V* default_values, int64_t default_value_num, bool* exists) = 0;
   virtual TFRA_Status Accum(const K* keys, int64_t key_num,
                             const V* values_or_delta, int64_t vod_num, 
                             const bool* exists) = 0;
   virtual TFRA_Status Clear() = 0;
   virtual TFRA_Status Dump(K* const key_buffer, V* const value_buffer, size_t search_offset, size_t buffer_size, size_t* dumped_counter) = 0;
  //  virtual TFRA_Status SaveToFileSystem(const std::string& dirpath, const std::string &file_name, const size_t buffer_size. bool append_to_file) = 0;
  //  virtual TFRA_Status LoadFromFileSystem(const std::string& dirpath, const std::string& file_name. const size_t buffer_size, bool load_entire_dir) = 0;
   virtual TFRA_DataType key_dtype() const = 0;
   virtual TFRA_DataType value_dtype() const = 0;
   virtual std::vector<int> key_shape() const = 0;
   virtual std::vector<int> value_shape() const = 0;
   virtual int64_t MemoryUsed() const = 0;
};   // class TFRALookupTableInterface

}   // namespace lookup_table
}   // namespace recommenders_addons
}   // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_LOOKUP_TABLE_INTERFACE_H_