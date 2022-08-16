/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <fstream>
#include <iostream>
#if __cplusplus >= 201703L
#include <filesystem>
#else
#include <sys/stat.h>
#endif
#include "../utils/utils.h"
#include "rocksdb/db.h"
#include "rocksdb_table_op.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace recommenders_addons {
namespace lookup_rocksdb {

static const size_t BATCH_SIZE_MIN = 2;
static const size_t BATCH_SIZE_MAX = 128;

static const uint32_t FILE_MAGIC =
    (  // TODO: Little endian / big endian conversion?
        (static_cast<uint32_t>('R') << 0) | (static_cast<uint32_t>('O') << 8) |
        (static_cast<uint32_t>('C') << 16) |
        (static_cast<uint32_t>('K') << 24));
static const uint32_t FILE_VERSION = 1;

typedef uint16_t KEY_SIZE_TYPE;
typedef uint32_t VALUE_SIZE_TYPE;
typedef uint32_t STRING_SIZE_TYPE;

#define ROCKSDB_OK(EXPR)                                                  \
  do {                                                                    \
    const ROCKSDB_NAMESPACE::Status s = (EXPR);                           \
    if (!s.ok()) {                                                        \
      std::ostringstream msg;                                             \
      msg << "RocksDB error " << s.code() << "; reason: " << s.getState() \
          << "; expr: " << #EXPR;                                         \
      throw std::runtime_error(msg.str());                                \
    }                                                                     \
  } while (0)

namespace _if {

template <class T>
inline void put_key(ROCKSDB_NAMESPACE::Slice &dst, const T *src) {
  dst.data_ = reinterpret_cast<const char *>(src);
  dst.size_ = sizeof(T);
}

template <>
inline void put_key<tstring>(ROCKSDB_NAMESPACE::Slice &dst,
                             const tstring *src) {
  dst.data_ = src->data();
  dst.size_ = src->size();
}

template <class T>
inline void get_value(T *dst, const std::string &src, const size_t &n) {
  const size_t dst_size = n * sizeof(T);

  if (src.size() < dst_size) {
    std::ostringstream msg;
    msg << "Expected " << n * sizeof(T) << " bytes, but only " << src.size()
        << " bytes were returned by the database.";
    throw std::runtime_error(msg.str());
  } else if (src.size() > dst_size) {
    LOG(WARNING) << "Expected " << dst_size << " bytes. The database returned "
                 << src.size() << ", which is more. Truncating!";
  }

  std::memcpy(dst, src.data(), dst_size);
}

template <>
inline void get_value<tstring>(tstring *dst, const std::string &src_,
                               const size_t &n) {
  const char *src = src_.data();
  const char *const src_end = &src[src_.size()];
  const tstring *const dst_end = &dst[n];

  for (; dst != dst_end; ++dst) {
    const char *const src_size = src;
    src += sizeof(STRING_SIZE_TYPE);
    if (src > src_end) {
      throw std::out_of_range("String value is malformed!");
    }
    const auto &size = *reinterpret_cast<const STRING_SIZE_TYPE *>(src_size);

    const char *const src_data = src;
    src += size;
    if (src > src_end) {
      throw std::out_of_range("String value is malformed!");
    }
    dst->assign(src_data, size);
  }

  if (src != src_end) {
    throw std::runtime_error(
        "Database returned more values than the destination tensor could "
        "absorb.");
  }
}

template <class T>
inline void put_value(ROCKSDB_NAMESPACE::PinnableSlice &dst, const T *src,
                      const size_t &n) {
  dst.data_ = reinterpret_cast<const char *>(src);
  dst.size_ = sizeof(T) * n;
}

template <>
inline void put_value<tstring>(ROCKSDB_NAMESPACE::PinnableSlice &dst_,
                               const tstring *src, const size_t &n) {
  std::string &dst = *dst_.GetSelf();
  dst.clear();

  // Concatenate the strings.
  const tstring *const src_end = &src[n];
  for (; src != src_end; ++src) {
    if (src->size() > std::numeric_limits<STRING_SIZE_TYPE>::max()) {
      throw std::runtime_error("String value is too large.");
    }
    const auto size = static_cast<STRING_SIZE_TYPE>(src->size());
    dst.append(reinterpret_cast<const char *>(&size), sizeof(size));
    dst.append(src->data(), size);
  }

  dst_.PinSelf();
}

}  // namespace _if

namespace _io {

template <class T>
inline void read(std::istream &src, T &dst) {
  if (!src.read(reinterpret_cast<char *>(&dst), sizeof(T))) {
    throw std::overflow_error("Unexpected end of file!");
  }
}

template <class T>
inline T read(std::istream &src) {
  T tmp;
  read(src, tmp);
  return tmp;
}

template <class T>
inline void write(std::ostream &dst, const T &src) {
  if (!dst.write(reinterpret_cast<const char *>(&src), sizeof(T))) {
    throw std::runtime_error("Writing file failed!");
  }
}

template <class T>
inline void read_key(std::istream &src, std::string *dst) {
  dst->resize(sizeof(T));
  if (!src.read(&dst->front(), sizeof(T))) {
    throw std::overflow_error("Unexpected end of file!");
  }
}

template <>
inline void read_key<tstring>(std::istream &src, std::string *dst) {
  const auto size = read<KEY_SIZE_TYPE>(src);
  dst->resize(size);
  if (!src.read(&dst->front(), size)) {
    throw std::overflow_error("Unexpected end of file!");
  }
}

template <class T>
inline void write_key(std::ostream &dst, const ROCKSDB_NAMESPACE::Slice &src) {
  write(dst, *reinterpret_cast<const T *>(src.data()));
}

template <>
inline void write_key<tstring>(std::ostream &dst,
                               const ROCKSDB_NAMESPACE::Slice &src) {
  if (src.size() > std::numeric_limits<KEY_SIZE_TYPE>::max()) {
    throw std::overflow_error("String key is too long for RDB_KEY_SIZE_TYPE.");
  }
  const auto size = static_cast<KEY_SIZE_TYPE>(src.size());
  write(dst, size);
  if (!dst.write(src.data(), size)) {
    throw std::runtime_error("Writing file failed!");
  }
}

inline void read_value(std::istream &src, std::string *dst) {
  const auto size = read<VALUE_SIZE_TYPE>(src);
  dst->resize(size);
  if (!src.read(&dst->front(), size)) {
    throw std::overflow_error("Unexpected end of file!");
  }
}

inline void write_value(std::ostream &dst,
                        const ROCKSDB_NAMESPACE::Slice &src) {
  const auto size = static_cast<VALUE_SIZE_TYPE>(src.size());
  write(dst, size);
  if (!dst.write(src.data(), size)) {
    throw std::runtime_error("Writing file failed!");
  }
}

}  // namespace _io

namespace _it {

template <class T>
inline void read_key(std::vector<T> &dst, const ROCKSDB_NAMESPACE::Slice &src) {
  if (src.size() != sizeof(T)) {
    std::ostringstream msg;
    msg << "Key size is out of bounds [ " << src.size() << " != " << sizeof(T)
        << " ].";
    throw std::out_of_range(msg.str());
  }
  dst.emplace_back(*reinterpret_cast<const T *>(src.data()));
}

template <>
inline void read_key<tstring>(std::vector<tstring> &dst,
                              const ROCKSDB_NAMESPACE::Slice &src) {
  if (src.size() > std::numeric_limits<KEY_SIZE_TYPE>::max()) {
    std::ostringstream msg;
    msg << "Key size is out of bounds [ " << src.size() << " > "
        << std::numeric_limits<KEY_SIZE_TYPE>::max() << " ].";
    throw std::out_of_range(msg.str());
  }
  dst.emplace_back(src.data(), src.size());
}

template <class T>
inline size_t read_value(std::vector<T> &dst,
                         const ROCKSDB_NAMESPACE::Slice &src_,
                         const size_t &n_limit) {
  const size_t n = src_.size() / sizeof(T);

  if (n * sizeof(T) != src_.size()) {
    std::ostringstream msg;
    msg << "Vector value is out of bounds [ " << n * sizeof(T)
        << " != " << src_.size() << " ].";
    throw std::out_of_range(msg.str());
  } else if (n < n_limit) {
    throw std::underflow_error("Database entry violates nLimit.");
  }

  const T *const src = reinterpret_cast<const T *>(src_.data());
  dst.insert(dst.end(), src, &src[n_limit]);
  return n;
}

template <>
inline size_t read_value<tstring>(std::vector<tstring> &dst,
                                  const ROCKSDB_NAMESPACE::Slice &src_,
                                  const size_t &n_limit) {
  size_t n = 0;

  const char *src = src_.data();
  const char *const src_end = &src[src_.size()];

  for (; src < src_end; ++n) {
    const char *const src_size = src;
    src += sizeof(STRING_SIZE_TYPE);
    if (src > src_end) {
      throw std::out_of_range("String value is malformed!");
    }
    const auto &size = *reinterpret_cast<const STRING_SIZE_TYPE *>(src_size);

    const char *const src_data = src;
    src += size;
    if (src > src_end) {
      throw std::out_of_range("String value is malformed!");
    }
    if (n < n_limit) {
      dst.emplace_back(src_data, size);
    }
  }

  if (src != src_end) {
    throw std::out_of_range("String value is malformed!");
  } else if (n < n_limit) {
    throw std::underflow_error("Database entry violates nLimit.");
  }
  return n;
}

}  // namespace _it

class DBWrapper final {
 public:
  DBWrapper(const std::string &path, const bool &read_only)
      : path_(path), read_only_(read_only), database_(nullptr) {
    ROCKSDB_NAMESPACE::Options options;
    options.create_if_missing = !read_only;
    options.manual_wal_flush = false;

    // Create or connect to the RocksDB database.
    std::vector<std::string> column_names;
#if __cplusplus >= 201703L
    if (!std::filesystem::exists(path)) {
      colFamilies.push_back(ROCKSDB_NAMESPACE::kDefaultColumnFamilyName);
    } else if (std::filesystem::is_directory(path)) {
      ROCKSDB_OK(ROCKSDB_NAMESPACE::DB::ListColumnFamilies(options, path,
                                                           &column_names));
    } else {
      throw std::runtime_error("Provided database path is invalid.");
    }
#else
    struct stat db_path_stat {};
    if (stat(path.c_str(), &db_path_stat) == 0) {
      if (S_ISDIR(db_path_stat.st_mode)) {
        ROCKSDB_OK(ROCKSDB_NAMESPACE::DB::ListColumnFamilies(options, path,
                                                             &column_names));
      } else {
        throw std::runtime_error("Provided database path is invalid.");
      }
    } else {
      column_names.push_back(ROCKSDB_NAMESPACE::kDefaultColumnFamilyName);
    }
#endif

    ROCKSDB_NAMESPACE::ColumnFamilyOptions column_options;
    std::vector<ROCKSDB_NAMESPACE::ColumnFamilyDescriptor> column_descriptors;
    for (const auto &column_name : column_names) {
      column_descriptors.emplace_back(column_name, column_options);
    }

    ROCKSDB_NAMESPACE::DB *db;
    std::vector<ROCKSDB_NAMESPACE::ColumnFamilyHandle *> column_handles;
    if (read_only) {
      ROCKSDB_OK(ROCKSDB_NAMESPACE::DB::OpenForReadOnly(
          options, path, column_descriptors, &column_handles, &db));
    } else {
      ROCKSDB_OK(ROCKSDB_NAMESPACE::DB::Open(options, path, column_descriptors,
                                             &column_handles, &db));
    }
    database_.reset(db);

    // Maintain map of the available column handles for quick access.
    for (const auto &column_handle : column_handles) {
      column_handles_[column_handle->GetName()] = column_handle;
    }

    LOG(INFO) << "Connected to database \'" << path_ << "\'.";
  }

  ~DBWrapper() {
    for (const auto &column_handle : column_handles_) {
      if (!read_only_) {
        database_->FlushWAL(true);
      }
      database_->DestroyColumnFamilyHandle(column_handle.second);
    }
    column_handles_.clear();
    database_.reset();
    LOG(INFO) << "Disconnected from database \'" << path_ << "\'.";
  }

  inline ROCKSDB_NAMESPACE::DB *database() { return database_.get(); }

  inline const std::string &path() const { return path_; }

  inline bool read_only() const { return read_only_; }

  void DeleteColumn(const std::string &column_name) {
    mutex_lock guard(lock_);

    // Try to locate column handle, and return if it anyway doe not exist.
    const auto &item = column_handles_.find(column_name);
    if (item == column_handles_.end()) {
      return;
    }

    // If a modification would be required make sure we are not in readonly
    // mode.
    if (read_only_) {
      throw std::runtime_error("Cannot delete a column in read-only mode.");
    }

    // Perform actual removal.
    ROCKSDB_NAMESPACE::ColumnFamilyHandle *column_handle = item->second;
    ROCKSDB_OK(database_->DropColumnFamily(column_handle));
    ROCKSDB_OK(database_->DestroyColumnFamilyHandle(column_handle));
    column_handles_.erase(column_name);
  }

  template <class T>
  T WithColumn(
      const std::string &column_name,
      std::function<T(ROCKSDB_NAMESPACE::DB* db, ROCKSDB_NAMESPACE::ColumnFamilyHandle *)> fn) {
    mutex_lock guard(lock_);

    ROCKSDB_NAMESPACE::ColumnFamilyHandle *column_handle;

    // Try to locate column handle.
    const auto &item = column_handles_.find(column_name);
    if (item != column_handles_.end()) {
      column_handle = item->second;
    }
    // Do not create an actual column handle in readonly mode.
    else if (read_only_) {
      column_handle = nullptr;
    }
    // Create a new column handle.
    else {
      ROCKSDB_NAMESPACE::ColumnFamilyOptions colFamilyOptions;
      ROCKSDB_OK(database_->CreateColumnFamily(colFamilyOptions, column_name,
                                               &column_handle));
      column_handles_[column_name] = column_handle;
    }

    return fn(database_.get(), column_handle);
  }

  // inline ROCKSDB_NAMESPACE::DB *operator->() { return database_.get(); }

 private:
  const std::string path_;
  const bool read_only_;
  std::unique_ptr<ROCKSDB_NAMESPACE::DB> database_;

  mutex lock_;
  std::unordered_map<std::string, ROCKSDB_NAMESPACE::ColumnFamilyHandle *>
      column_handles_;
};

class DBWrapperRegistry final {
 public:
  static DBWrapperRegistry &instance() {
    static DBWrapperRegistry instance;
    return instance;
  }

 private:
  DBWrapperRegistry() = default;

  ~DBWrapperRegistry() = default;

 public:
  std::shared_ptr<DBWrapper> connect(const std::string &databasePath,
                                     const bool &readOnly) {
    mutex_lock guard(lock);

    // Try to find database, or open it if it is not open yet.
    std::shared_ptr<DBWrapper> db;
    auto pos = wrappers.find(databasePath);
    if (pos != wrappers.end()) {
      db = pos->second.lock();
    } else {
      db.reset(new DBWrapper(databasePath, readOnly), deleter);
      wrappers[databasePath] = db;
    }

    // Suicide, if the desired access level is below the available access level.
    if (readOnly < db->read_only()) {
      throw std::runtime_error(
          "Cannot simultaneously open database in read + write mode.");
    }

    return db;
  }

 private:
  static void deleter(DBWrapper *wrapper) {
    static std::default_delete<DBWrapper> default_deleter;

    DBWrapperRegistry &registry = instance();
    const std::string path = wrapper->path();

    // Make sure we are alone.
    mutex_lock guard(registry.lock);

    // Destroy the wrapper.
    default_deleter(wrapper);
    // LOG(INFO) << "Database wrapper " << path << " has been deleted.";

    // Locate the corresponding weak_ptr and evict it.
    auto pos = registry.wrappers.find(path);
    if (pos == registry.wrappers.end()) {
      LOG(ERROR) << "Unknown database wrapper. How?";
    } else if (pos->second.expired()) {
      registry.wrappers.erase(pos);
      // LOG(INFO) << "Database wrapper " << path << " evicted.";
    } else {
      LOG(ERROR) << "Registry is in an inconsistent state. This is very bad...";
    }
  }

 private:
  mutex lock;
  std::unordered_map<std::string, std::weak_ptr<DBWrapper>> wrappers;
};

template <class K, class V>
class RocksDBTableOfTensors final : public PersistentStorageLookupInterface {
 public:
  /* --- BASE INTERFACE ----------------------------------------------------- */
  RocksDBTableOfTensors(OpKernelContext *ctx, OpKernel *kernel)
      : read_only_(false), estimate_size_(false), dirty_count_(0) {
    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(kernel->def(), "value_shape", &value_shape_));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(value_shape_),
        errors::InvalidArgument("Default value must be a vector, got shape ",
                                value_shape_.DebugString()));
    OP_REQUIRES_OK(
        ctx, GetNodeAttr(kernel->def(), "database_path", &database_path_));
    OP_REQUIRES_OK(
        ctx, GetNodeAttr(kernel->def(), "embedding_name", &embedding_name_));
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "read_only", &read_only_));
    OP_REQUIRES_OK(
        ctx, GetNodeAttr(kernel->def(), "estimate_size", &estimate_size_));
    flush_interval_ = 1;
    OP_REQUIRES_OK(
        ctx, GetNodeAttr(kernel->def(), "export_path", &default_export_path_));

    db_ = DBWrapperRegistry::instance().connect(database_path_, read_only_);
    LOG(INFO) << "Acquired reference to database wrapper " << db_->path()
              << " [ #refs = " << db_.use_count() << " ].";
  }

  ~RocksDBTableOfTensors() override {
    LOG(INFO) << "Dropping reference to database wrapper " << db_->path()
              << " [ #refs = " << db_.use_count() << " ].";
  }

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }
  TensorShape key_shape() const override { return TensorShape{}; }

  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }
  TensorShape value_shape() const override { return value_shape_; }

  int64_t MemoryUsed() const override {
    size_t mem_size = 0;
    
    mem_size += sizeof(RocksDBTableOfTensors);
    mem_size += sizeof(ROCKSDB_NAMESPACE::DB);

    db_->WithColumn<void>(embedding_name_, [&](ROCKSDB_NAMESPACE::DB* const db, ROCKSDB_NAMESPACE::ColumnFamilyHandle *const column_handle) {
      uint64_t tmp;

      if (db->GetIntProperty(column_handle, ROCKSDB_NAMESPACE::DB::Properties::kBlockCacheUsage, &tmp)) {
        mem_size += tmp;
      }
      
      
      if (db->GetIntProperty(column_handle, ROCKSDB_NAMESPACE::DB::Properties::kEstimateTableReadersMem, &tmp)) {
        mem_size += tmp;
      }
      
      if (db->GetIntProperty(column_handle, ROCKSDB_NAMESPACE::DB::Properties::kCurSizeAllMemTables, &tmp)) {
        mem_size += tmp;
      }
      
      if (db->GetIntProperty(column_handle, ROCKSDB_NAMESPACE::DB::Properties::kBlockCachePinnedUsage, &tmp)) {
        mem_size += tmp;
      }
    });
    
    return static_cast<int64_t>(mem_size);
  }

  size_t size() const override {
    return db_->WithColumn<size_t>(embedding_name_, [&](ROCKSDB_NAMESPACE::DB* const db, ROCKSDB_NAMESPACE::ColumnFamilyHandle *const column_handle)
        -> size_t {
      // Empty database.
      if (!column_handle) {
        return 0;
      }

      // If allowed, try to just estimate of the number of keys.
      if (estimate_size_) {
        uint64_t num_keys;
        if (db->GetIntProperty(
                column_handle,
                ROCKSDB_NAMESPACE::DB::Properties::kEstimateNumKeys,
                &num_keys)) {
          return num_keys;
        }
      }

      // Alternative method, walk the entire database column and count the keys.
      std::unique_ptr<ROCKSDB_NAMESPACE::Iterator> iter(
          db->NewIterator(read_options_, column_handle));
      iter->SeekToFirst();

      size_t num_keys = 0;
      for (; iter->Valid(); iter->Next()) {
        ++num_keys;
      }
      return num_keys;
    });
  }

  /* --- LOOKUP ------------------------------------------------------------- */
  /*
  Status Accum(OpKernelContext *ctx, const Tensor &keys,
               const Tensor &values_or_delta, const Tensor &exists) {
                
               }
  */

  Status Clear(OpKernelContext *ctx) override {
    if (read_only_) {
      return errors::PermissionDenied("Cannot clear in read_only mode.");
    }
    db_->DeleteColumn(embedding_name_);
    return Status::OK();
  }

  Status Find(OpKernelContext *ctx, const Tensor &keys, Tensor *values,
              const Tensor &default_value) override {
    if (keys.dtype() != key_dtype() || values->dtype() != value_dtype() ||
        default_value.dtype() != value_dtype()) {
      return errors::InvalidArgument("The tensor dtypes are incompatible.");
    }
    if (keys.dims() > values->dims()) {
      return errors::InvalidArgument("The tensor sizes are incompatible.");
    }
    for (int i = 0; i < keys.dims(); ++i) {
      if (keys.dim_size(i) != values->dim_size(i)) {
        return errors::InvalidArgument("The tensor sizes are incompatible.");
      }
    }
    if (keys.NumElements() == 0) {
      return Status::OK();
    }

    const size_t num_keys = keys.NumElements();
    const size_t num_values = values->NumElements();
    const size_t values_per_key = num_values / std::max(num_keys, 1UL);
    const size_t default_size = default_value.NumElements();
    if (default_size % values_per_key != 0) {
      std::ostringstream msg;
      msg << "The shapes of the 'values' and 'default_value' tensors are "
             "incompatible"
          << " (" << default_size << " % " << values_per_key << " != 0).";
      return errors::InvalidArgument(msg.str());
    }

    const K *k = static_cast<K *>(keys.data());
    V *const v = static_cast<V *>(values->data());
    const V *const d = static_cast<V *>(default_value.data());

    return db_->WithColumn<Status>(embedding_name_, [&](ROCKSDB_NAMESPACE::DB* const db, ROCKSDB_NAMESPACE::ColumnFamilyHandle *const column_handle)
        -> Status {
      if (!column_handle) {
        const K *const k_end = &k[num_keys];
        for (size_t offset = 0; k != k_end; ++k, offset += values_per_key) {
          std::copy_n(&d[offset % default_size], values_per_key, &v[offset]);
        }
      } else if (num_keys < BATCH_SIZE_MIN) {
        ROCKSDB_NAMESPACE::Slice k_slice;

        std::string v_slice;
        for (size_t i = 0, offset = 0; i < num_keys; ++i, offset += values_per_key) {
          _if::put_key(k_slice, &k[i]);

          v_slice.clear();
          const auto &status =
              db->Get(read_options_, column_handle, k_slice, &v_slice);

          if (status.ok()) {
            _if::get_value(&v[offset], v_slice, values_per_key);
          } else if (status.IsNotFound()) {
            std::copy_n(&d[offset % default_size], values_per_key, &v[offset]);
          } else {
            throw std::runtime_error(status.getState());
          }
        }
      } else {
        // There is no point in filling this vector every time as long as it is
        // big enough.
        if (!column_handle_cache_.empty() &&
            column_handle_cache_.front() != column_handle) {
          std::fill(column_handle_cache_.begin(), column_handle_cache_.end(),
                    column_handle);
        }
        if (column_handle_cache_.size() < num_keys) {
          column_handle_cache_.insert(column_handle_cache_.end(),
                                      num_keys - column_handle_cache_.size(),
                                      column_handle);
        }

        // Query all keys using a single Multi-Get.
        std::vector<ROCKSDB_NAMESPACE::Slice> k_slices{num_keys};
        for (size_t i = 0; i < num_keys; ++i) {
          _if::put_key(k_slices[i], &k[i]);
        }
        std::vector<std::string> v_slices;

        const auto &s = db->MultiGet(read_options_, column_handle_cache_,
                                         k_slices, &v_slices);
        if (s.size() != num_keys) {
          std::ostringstream msg;
          msg << "Requested " << num_keys << " keys, but only got " << s.size()
              << " responses.";
          throw std::runtime_error(msg.str());
        }

        // Process results.
        for (size_t i = 0, offset = 0; i < num_keys;
             ++i, offset += values_per_key) {
          const auto &status = s[i];
          const auto &v_slice = v_slices[i];

          if (status.ok()) {
            _if::get_value(&v[offset], v_slice, values_per_key);
          } else if (status.IsNotFound()) {
            std::copy_n(&d[offset % default_size], values_per_key, &v[offset]);
          } else {
            throw std::runtime_error(status.getState());
          }
        }
      }

      return Status::OK();
    });
  }
  
  Status FindWithExists(OpKernelContext *ctx, const Tensor &keys,
                        Tensor *values, const Tensor &default_value,
                        Tensor &exists) {
    if (keys.dtype() != key_dtype() || values->dtype() != value_dtype() ||
        default_value.dtype() != value_dtype()) {
      return errors::InvalidArgument("The tensor dtypes are incompatible.");
    }
    if (keys.dims() > values->dims()) {
      return errors::InvalidArgument("The tensor sizes are incompatible.");
    }
    for (int i = 0; i < keys.dims(); ++i) {
      if (keys.dim_size(i) != values->dim_size(i) || keys.dim_size(i) != exists.dim_size(i)) {
        return errors::InvalidArgument("The tensor sizes are incompatible.");
      }
    }
    if (keys.NumElements() == 0) {
      return Status::OK();
    }

    const size_t num_keys = keys.NumElements();
    const size_t num_values = values->NumElements();
    const size_t values_per_key = num_values / std::max(num_keys, 1UL);
    const size_t default_size = default_value.NumElements();
    if (default_size % values_per_key != 0) {
      std::ostringstream msg;
      msg << "The shapes of the 'values' and 'default_value' tensors are "
             "incompatible"
          << " (" << default_size << " % " << values_per_key << " != 0).";
      return errors::InvalidArgument(msg.str());
    }

    const K *k = static_cast<K *>(keys.data());
    V *const v = static_cast<V *>(values->data());
    const V *const d = static_cast<V *>(default_value.data());
    auto exists_flat = exists.flat<bool>();

    return db_->WithColumn<Status>(embedding_name_, [&](ROCKSDB_NAMESPACE::DB* const db, ROCKSDB_NAMESPACE::ColumnFamilyHandle *const column_handle)
        -> Status {
      if (!column_handle) {
        const K *const k_end = &k[num_keys];
        for (size_t offset = 0; k != k_end; ++k, offset += values_per_key) {
          std::copy_n(&d[offset % default_size], values_per_key, &v[offset]);
        }
      } else if (num_keys < BATCH_SIZE_MIN) {
        ROCKSDB_NAMESPACE::Slice k_slice;

        std::string v_slice;
        for (size_t i = 0, offset = 0; i < num_keys; ++i, offset += values_per_key) {
          _if::put_key(k_slice, &k[i]);
          
          v_slice.clear();
          const auto &status =
              db->Get(read_options_, column_handle, k_slice, &v_slice);

          if (status.ok()) {
            _if::get_value(&v[offset], v_slice, values_per_key);
            exists_flat(i) = true;
          } else if (status.IsNotFound()) {
            std::copy_n(&d[offset % default_size], values_per_key, &v[offset]);
            exists_flat(i) = false;
          } else {
            throw std::runtime_error(status.getState());
          }
        }
      } else {
        // There is no point in filling this vector every time as long as it is
        // big enough.
        if (!column_handle_cache_.empty() &&
            column_handle_cache_.front() != column_handle) {
          std::fill(column_handle_cache_.begin(), column_handle_cache_.end(),
                    column_handle);
        }
        if (column_handle_cache_.size() < num_keys) {
          column_handle_cache_.insert(column_handle_cache_.end(),
                                      num_keys - column_handle_cache_.size(),
                                      column_handle);
        }

        // Query all keys using a single Multi-Get.
        std::vector<ROCKSDB_NAMESPACE::Slice> k_slices{num_keys};
        for (size_t i = 0; i < num_keys; ++i) {
          _if::put_key(k_slices[i], &k[i]);
        }
        std::vector<std::string> v_slices;

        const auto &s = db->MultiGet(read_options_, column_handle_cache_,
                                         k_slices, &v_slices);
        if (s.size() != num_keys) {
          std::ostringstream msg;
          msg << "Requested " << num_keys << " keys, but only got " << s.size()
              << " responses.";
          throw std::runtime_error(msg.str());
        }

        // Process results.
        for (size_t i = 0, offset = 0; i < num_keys;
             ++i, offset += values_per_key) {
          const auto &status = s[i];
          const auto &v_slice = v_slices[i];

          if (status.ok()) {
            _if::get_value(&v[offset], v_slice, values_per_key);
            exists_flat(i) = true;
          } else if (status.IsNotFound()) {
            std::copy_n(&d[offset % default_size], values_per_key, &v[offset]);
            exists_flat(i) = false;
          } else {
            throw std::runtime_error(status.getState());
          }
        }
      }

      return Status::OK();
    });
  }

  Status Insert(OpKernelContext *ctx, const Tensor &keys,
                const Tensor &values) override {
    if (keys.dtype() != key_dtype() || values.dtype() != value_dtype()) {
      return errors::InvalidArgument("The tensor dtypes are incompatible!");
    }
    if (keys.dims() <= values.dims()) {
      for (int i = 0; i < keys.dims(); ++i) {
        if (keys.dim_size(i) != values.dim_size(i)) {
          return errors::InvalidArgument("The tensor sizes are incompatible!");
        }
      }
    } else {
      return errors::InvalidArgument("The tensor sizes are incompatible!");
    }

    const size_t num_keys = keys.NumElements();
    const size_t num_values = values.NumElements();
    const size_t values_per_key = num_values / std::max(num_keys, 1UL);
    if (values_per_key != static_cast<size_t>(value_shape_.num_elements())) {
      LOG(WARNING)
          << "The number of values provided does not match the signature ("
          << values_per_key << " != " << value_shape_.num_elements() << ").";
    }

    const K *k = static_cast<K *>(keys.data());
    const V *v = static_cast<V *>(values.data());

    return db_->WithColumn<Status>(embedding_name_, [&](ROCKSDB_NAMESPACE::DB* const db, ROCKSDB_NAMESPACE::ColumnFamilyHandle *const column_handle)
        -> Status {
      if (read_only_ || !column_handle) {
        return errors::PermissionDenied("Cannot insert in read_only mode.");
      }

      const K *const k_end = &k[num_keys];
      ROCKSDB_NAMESPACE::Slice k_slice;
      ROCKSDB_NAMESPACE::PinnableSlice v_slice;

      if (num_keys < BATCH_SIZE_MIN) {
        for (; k != k_end; ++k, v += values_per_key) {
          _if::put_key(k_slice, k);
          _if::put_value(v_slice, v, values_per_key);
          ROCKSDB_OK(
              db->Put(write_options_, column_handle, k_slice, v_slice));
        }
      } else {
        ROCKSDB_NAMESPACE::WriteBatch batch;
        for (; k != k_end; ++k, v += values_per_key) {
          _if::put_key(k_slice, k);
          _if::put_value(v_slice, v, values_per_key);
          ROCKSDB_OK(batch.Put(column_handle, k_slice, v_slice));
        }
        ROCKSDB_OK(db->Write(write_options_, &batch));
      }

      // Handle interval flushing.
      dirty_count_ += 1;
      if (dirty_count_ % flush_interval_ == 0) {
        ROCKSDB_OK(db->FlushWAL(true));
      }

      return Status::OK();
    });
  }

  Status Remove(OpKernelContext *ctx, const Tensor &keys) override {
    if (keys.dtype() != key_dtype()) {
      return errors::InvalidArgument("Tensor dtypes are incompatible!");
    }

    const size_t num_keys = keys.dim_size(0);
    const K *k = static_cast<K *>(keys.data());

    return db_->WithColumn<Status>(embedding_name_, [&](ROCKSDB_NAMESPACE::DB* const db, ROCKSDB_NAMESPACE::ColumnFamilyHandle *const column_handle)
        -> Status {
      if (read_only_ || !column_handle) {
        return errors::PermissionDenied("Cannot remove in read_only mode.");
      }

      const K *const k_end = &k[num_keys];
      ROCKSDB_NAMESPACE::Slice k_slice;

      if (num_keys < BATCH_SIZE_MIN) {
        for (; k != k_end; ++k) {
          _if::put_key(k_slice, k);
          ROCKSDB_OK(db->Delete(write_options_, column_handle, k_slice));
        }
      } else {
        ROCKSDB_NAMESPACE::WriteBatch batch;
        for (; k != k_end; ++k) {
          _if::put_key(k_slice, k);
          ROCKSDB_OK(batch.Delete(column_handle, k_slice));
        }
        ROCKSDB_OK(db->Write(write_options_, &batch));
      }

      // Handle interval flushing.
      dirty_count_ += 1;
      if (dirty_count_ % flush_interval_ == 0) {
        ROCKSDB_OK(db->FlushWAL(true));
      }

      return Status::OK();
    });
  }

  /* --- IMPORT / EXPORT ---------------------------------------------------- */
  Status ExportValues(OpKernelContext *ctx) override {
    if (default_export_path_.empty()) {
      return ExportValuesToTensor(ctx);
    } else {
      return ExportValuesToFile(ctx, default_export_path_);
    }
  }
  Status ImportValues(OpKernelContext *ctx, const Tensor &keys,
                      const Tensor &values) override {
    if (default_export_path_.empty()) {
      return ImportValuesFromTensor(ctx, keys, values);
    } else {
      return ImportValuesFromFile(ctx, default_export_path_);
    }
  }

  Status ExportValuesToFile(OpKernelContext *ctx, const std::string &path) {
    const auto &status = db_->WithColumn<Status>(embedding_name_, [&](ROCKSDB_NAMESPACE::DB *const db, ROCKSDB_NAMESPACE::ColumnFamilyHandle *const column_handle)
        -> Status {
      std::ofstream file(path + "/" + embedding_name_ + ".rock",
                         std::ofstream::binary);
      if (!file) {
        return errors::Unknown("Could not open dump file.");
      }

      // Create file header.
      _io::write(file, FILE_MAGIC);
      _io::write(file, FILE_VERSION);
      _io::write(file, key_dtype());
      _io::write(file, value_dtype());

      // Iterate through entries one-by-one and append them to the file.
      if (column_handle) {
        std::unique_ptr<ROCKSDB_NAMESPACE::Iterator> iter(
            db->NewIterator(read_options_, column_handle));
        iter->SeekToFirst();

        for (; iter->Valid(); iter->Next()) {
          _io::write_key<K>(file, iter->key());
          _io::write_value(file, iter->value());
        }
      }

      return Status::OK();
    });
    if (!status.ok()) {
      return status;
    }

    // Creat dummy tensors.
    Tensor *k_tensor;
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({0}), &k_tensor));

    Tensor *v_tensor;
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "values", TensorShape({0, value_shape_.num_elements()}), &v_tensor));

    return status;
  }
  Status ImportValuesFromFile(OpKernelContext *ctx, const std::string &path) {
    // Make sure the column family is clean.
    const auto &clear_status = Clear(ctx);
    if (!clear_status.ok()) {
      return clear_status;
    }

    return db_->WithColumn<Status>(embedding_name_, [&](ROCKSDB_NAMESPACE::DB *const db, ROCKSDB_NAMESPACE::ColumnFamilyHandle *const column_handle)
        -> Status {
      if (read_only_ || !column_handle) {
        return errors::PermissionDenied("Cannot import in read_only mode.");
      }

      std::ifstream file(path + "/" + embedding_name_ + ".rock",
                         std::ifstream::binary);
      if (!file) {
        return errors::NotFound("Accessing file system failed.");
      }

      // Parse header.
      const auto magic = _io::read<uint32_t>(file);
      if (magic != FILE_MAGIC) {
        return errors::Unknown("Not a RocksDB export file.");
      }
      const auto version = _io::read<uint32_t>(file);
      if (version != FILE_VERSION) {
        return errors::Unimplemented("File version ", version,
                                     " is not supported");
      }
      const auto k_dtype = _io::read<DataType>(file);
      const auto v_dtype = _io::read<DataType>(file);
      if (k_dtype != key_dtype() || v_dtype != value_dtype()) {
        return errors::Internal("DataType of file [k=", k_dtype,
                                ", v=", v_dtype, "] ",
                                "do not match module DataType [k=", key_dtype(),
                                ", v=", value_dtype(), "].");
      }

      // Read payload and subsequently populate column family.
      ROCKSDB_NAMESPACE::WriteBatch batch;

      ROCKSDB_NAMESPACE::PinnableSlice k_slice;
      ROCKSDB_NAMESPACE::PinnableSlice v_slice;

      while (file.peek() != EOF) {
        _io::read_key<K>(file, k_slice.GetSelf());
        k_slice.PinSelf();
        _io::read_value(file, v_slice.GetSelf());
        v_slice.PinSelf();

        ROCKSDB_OK(batch.Put(column_handle, k_slice, v_slice));

        // If batch reached target size, write to database.
        if (batch.Count() >= BATCH_SIZE_MAX) {
          ROCKSDB_OK(db->Write(write_options_, &batch));
          batch.Clear();
        }
      }

      // Write remaining entries, if any.
      if (batch.Count()) {
        ROCKSDB_OK(db->Write(write_options_, &batch));
      }

      // Handle interval flushing.
      dirty_count_ += 1;
      if (dirty_count_ % flush_interval_ == 0) {
        ROCKSDB_OK(db->FlushWAL(true));
      }

      return Status::OK();
    });
  }

  Status ExportValuesToTensor(OpKernelContext *ctx) {
    // Fetch data from database.
    std::vector<K> k_buffer;
    std::vector<V> v_buffer;
    const size_t value_size = value_shape_.num_elements();
    size_t value_count = std::numeric_limits<size_t>::max();

    const auto &status = db_->WithColumn<Status>(embedding_name_, [&](ROCKSDB_NAMESPACE::DB* const db, ROCKSDB_NAMESPACE::ColumnFamilyHandle *const column_handle)
        -> Status {
      if (column_handle) {
        std::unique_ptr<ROCKSDB_NAMESPACE::Iterator> iter(
            db->NewIterator(read_options_, column_handle));
        iter->SeekToFirst();

        for (; iter->Valid(); iter->Next()) {
          const auto &k_slice = iter->key();
          _it::read_key(k_buffer, k_slice);

          const auto v_slice = iter->value();
          const size_t v_count = _it::read_value(v_buffer, v_slice, value_size);

          // Make sure we have a square tensor.
          if (value_count == std::numeric_limits<size_t>::max()) {
            value_count = v_count;
          } else if (v_count != value_count) {
            return errors::Internal("The returned tensor sizes differ.");
          }
        }
      }

      return Status::OK();
    });
    if (!status.ok()) {
      return status;
    }

    if (value_count != value_size) {
      LOG(WARNING) << "Retrieved values differ from signature size ("
                   << value_count << " != " << value_size << ").";
    }
    const auto numKeys = static_cast<int64>(k_buffer.size());

    // Populate keys tensor.
    Tensor *k_tensor;
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({numKeys}), &k_tensor));
    K *const k = reinterpret_cast<K *>(k_tensor->data());
    std::copy(k_buffer.begin(), k_buffer.end(), k);

    // Populate values tensor.
    Tensor *v_tensor;
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "values", TensorShape({numKeys, static_cast<int64>(value_size)}),
        &v_tensor));
    V *const v = reinterpret_cast<V *>(v_tensor->data());
    std::copy(v_buffer.begin(), v_buffer.end(), v);

    return status;
  }
  Status ImportValuesFromTensor(OpKernelContext *ctx, const Tensor &keys,
                                const Tensor &values) {
    // Make sure the column family is clean.
    const auto &clear_status = Clear(ctx);
    if (!clear_status.ok()) {
      return clear_status;
    }

    // Just call normal insertion function.
    return Insert(ctx, keys, values);
  }

 protected:
  TensorShape value_shape_;
  std::string database_path_;
  std::string embedding_name_;
  bool read_only_;
  bool estimate_size_;
  size_t flush_interval_;
  std::string default_export_path_;

  std::shared_ptr<DBWrapper> db_;
  ROCKSDB_NAMESPACE::ReadOptions read_options_;
  ROCKSDB_NAMESPACE::WriteOptions write_options_;
  size_t dirty_count_;

  std::vector<ROCKSDB_NAMESPACE::ColumnFamilyHandle *> column_handle_cache_;
};

#undef ROCKSDB_OK

/* --- KERNEL REGISTRATION -------------------------------------------------- */
#define ROCKSDB_REGISTER_KERNEL_BUILDER(key_dtype, value_dtype)                \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(PREFIX_OP_NAME(RocksdbTableOfTensors))                              \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<key_dtype>("key_dtype")                              \
          .TypeConstraint<value_dtype>("value_dtype"),                         \
      RocksDBTableOp<RocksDBTableOfTensors<key_dtype, value_dtype>, key_dtype, \
                     value_dtype>)

ROCKSDB_REGISTER_KERNEL_BUILDER(int32, bool);
ROCKSDB_REGISTER_KERNEL_BUILDER(int32, int8);
ROCKSDB_REGISTER_KERNEL_BUILDER(int32, int16);
ROCKSDB_REGISTER_KERNEL_BUILDER(int32, int32);
ROCKSDB_REGISTER_KERNEL_BUILDER(int32, int64);
ROCKSDB_REGISTER_KERNEL_BUILDER(int32, Eigen::half);
ROCKSDB_REGISTER_KERNEL_BUILDER(int32, float);
ROCKSDB_REGISTER_KERNEL_BUILDER(int32, double);
ROCKSDB_REGISTER_KERNEL_BUILDER(int32, tstring);

ROCKSDB_REGISTER_KERNEL_BUILDER(int64, bool);
ROCKSDB_REGISTER_KERNEL_BUILDER(int64, int8);
ROCKSDB_REGISTER_KERNEL_BUILDER(int64, int16);
ROCKSDB_REGISTER_KERNEL_BUILDER(int64, int32);
ROCKSDB_REGISTER_KERNEL_BUILDER(int64, int64);
ROCKSDB_REGISTER_KERNEL_BUILDER(int64, Eigen::half);
ROCKSDB_REGISTER_KERNEL_BUILDER(int64, float);
ROCKSDB_REGISTER_KERNEL_BUILDER(int64, double);
ROCKSDB_REGISTER_KERNEL_BUILDER(int64, tstring);

ROCKSDB_REGISTER_KERNEL_BUILDER(tstring, bool);
ROCKSDB_REGISTER_KERNEL_BUILDER(tstring, int8);
ROCKSDB_REGISTER_KERNEL_BUILDER(tstring, int16);
ROCKSDB_REGISTER_KERNEL_BUILDER(tstring, int32);
ROCKSDB_REGISTER_KERNEL_BUILDER(tstring, int64);
ROCKSDB_REGISTER_KERNEL_BUILDER(tstring, Eigen::half);
ROCKSDB_REGISTER_KERNEL_BUILDER(tstring, float);
ROCKSDB_REGISTER_KERNEL_BUILDER(tstring, double);
ROCKSDB_REGISTER_KERNEL_BUILDER(tstring, tstring);

#undef ROCKSDB_REGISTER_KERNEL_BUILDER
}  // namespace lookup_rocksdb

/* --- OP KERNELS ----------------------------------------------------------- */
class RocksDBTableOpKernel : public OpKernel {
 public:
  explicit RocksDBTableOpKernel(OpKernelConstruction *ctx)
      : OpKernel(ctx),
        expected_input_0_(ctx->input_type(0) == DT_RESOURCE ? DT_RESOURCE
                                                            : DT_STRING_REF) {}

 protected:
  Status LookupResource(OpKernelContext *ctx, const ResourceHandle &p,
                        LookupInterface **value) {
    return ctx->resource_manager()->Lookup<LookupInterface, false>(
        p.container(), p.name(), value);
  }

  Status GetTableHandle(StringPiece input_name, OpKernelContext *ctx,
                        tstring *container, tstring *table_handle) {
    {
      mutex *guard;
      TF_RETURN_IF_ERROR(ctx->input_ref_mutex(input_name, &guard));
      mutex_lock lock(*guard);
      Tensor tensor;
      TF_RETURN_IF_ERROR(ctx->mutable_input(input_name, &tensor, true));
      if (tensor.NumElements() != 2) {
        return errors::InvalidArgument(
            "Lookup table handle must be scalar, but had shape: ",
            tensor.shape().DebugString());
      }
      auto h = tensor.flat<tstring>();
      *container = h(0);
      *table_handle = h(1);
    }
    return Status::OK();
  }

  Status GetResourceHashTable(StringPiece input_name, OpKernelContext *ctx,
                              LookupInterface **table) {
    const Tensor *handle_tensor;
    TF_RETURN_IF_ERROR(ctx->input(input_name, &handle_tensor));
    const auto &handle = handle_tensor->scalar<ResourceHandle>()();
    return LookupResource(ctx, handle, table);
  }

  Status GetReferenceLookupTable(StringPiece input_name, OpKernelContext *ctx,
                                 LookupInterface **table) {
    tstring container;
    tstring table_handle;
    TF_RETURN_IF_ERROR(
        GetTableHandle(input_name, ctx, &container, &table_handle));
    return ctx->resource_manager()->Lookup(container, table_handle, table);
  }

  Status GetTable(OpKernelContext *ctx, LookupInterface **table) {
    if (expected_input_0_ == DT_RESOURCE) {
      return GetResourceHashTable("table_handle", ctx, table);
    } else {
      return GetReferenceLookupTable("table_handle", ctx, table);
    }
  }

 protected:
  const DataType expected_input_0_;
};

class RocksDBTableClear : public RocksDBTableOpKernel {
 public:
  using RocksDBTableOpKernel::RocksDBTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    auto *rocks_table = dynamic_cast<PersistentStorageLookupInterface *>(table);

    int64 memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, rocks_table->Clear(ctx));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

class RocksDBTableExport : public RocksDBTableOpKernel {
 public:
  using RocksDBTableOpKernel::RocksDBTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, table->ExportValues(ctx));
  }
};

class RocksDBTableFind : public RocksDBTableOpKernel {
 public:
  using RocksDBTableOpKernel::RocksDBTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor &key = ctx->input(1);
    const Tensor &default_value = ctx->input(2);

    TensorShape output_shape = key.shape();
    output_shape.RemoveLastDims(table->key_shape().dims());
    output_shape.AppendShape(table->value_shape());
    Tensor *out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("values", output_shape, &out));
    OP_REQUIRES_OK(ctx, table->Find(ctx, key, out, default_value));
  }
};

class RocksDBTableImport : public RocksDBTableOpKernel {
 public:
  using RocksDBTableOpKernel::RocksDBTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                      table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor &keys = ctx->input(1);
    const Tensor &values = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForImport(keys, values));

    int64 memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, table->ImportValues(ctx, keys, values));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

class RocksDBTableInsert : public RocksDBTableOpKernel {
 public:
  using RocksDBTableOpKernel::RocksDBTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                      table->value_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor &keys = ctx->input(1);
    const Tensor &values = ctx->input(2);
    OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForInsert(keys, values));

    int64 memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, table->Insert(ctx, keys, values));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

class RocksDBTableRemove : public RocksDBTableOpKernel {
 public:
  using RocksDBTableOpKernel::RocksDBTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor &key = ctx->input(1);
    OP_REQUIRES_OK(ctx, table->CheckKeyTensorForRemove(key));

    int64 memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, table->Remove(ctx, key));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

class RocksDBTableSize : public RocksDBTableOpKernel {
 public:
  using RocksDBTableOpKernel::RocksDBTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    Tensor *out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("size", TensorShape({}), &out));
    out->flat<int64>().setConstant(static_cast<int64>(table->size()));
  }
};

REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(RocksdbTableClear)).Device(DEVICE_CPU),
    RocksDBTableClear);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(RocksdbTableExport)).Device(DEVICE_CPU),
    RocksDBTableExport);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(RocksdbTableFind)).Device(DEVICE_CPU),
    RocksDBTableFind);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(RocksdbTableImport)).Device(DEVICE_CPU),
    RocksDBTableImport);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(RocksdbTableInsert)).Device(DEVICE_CPU),
    RocksDBTableInsert);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(RocksdbTableRemove)).Device(DEVICE_CPU),
    RocksDBTableRemove);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(RocksdbTableSize)).Device(DEVICE_CPU),
    RocksDBTableSize);

}  // namespace recommenders_addons
}  // namespace tensorflow
