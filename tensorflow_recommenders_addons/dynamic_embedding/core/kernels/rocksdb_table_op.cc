/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <iostream>
#include <fstream>
#if __cplusplus >= 201703L
#include <filesystem>
#else
#include <sys/stat.h>
#endif
#include "../utils/utils.h"
#include "rocksdb_table_op.h"
#include "rocksdb/db.h"

namespace tensorflow {
  namespace recommenders_addons {
    namespace lookup_rocksdb {

      static const size_t BATCH_SIZE_MIN = 2;
      static const size_t BATCH_SIZE_MAX = 128;

      static const uint32_t FILE_MAGIC = (  // TODO: Little endian / big endian conversion?
        (static_cast<uint32_t>('T') <<  0) |
        (static_cast<uint32_t>('F') <<  8) |
        (static_cast<uint32_t>('K') << 16) |
        (static_cast<uint32_t>('V') << 24)
      );
      static const uint32_t FILE_VERSION = 1;

      typedef uint16_t KEY_SIZE_TYPE;
      typedef uint32_t VALUE_SIZE_TYPE;
      typedef uint32_t STRING_SIZE_TYPE;

      #define ROCKSDB_OK(EXPR)                                \
         do {                                                 \
            const ROCKSDB_NAMESPACE::Status s = EXPR;         \
            if (!s.ok()) {                                    \
              std::stringstream msg(std::stringstream::out);  \
              msg << "RocksDB error " << s.code()             \
                  << "; reason: " << s.getState()             \
                  << "; expr: " << #EXPR;                     \
              throw std::runtime_error(msg.str());            \
            }                                                 \
         } while (0)

      namespace _if {

        template<class T>
        inline void putKey(ROCKSDB_NAMESPACE::Slice &dst, const T *src) {
          dst.data_ = reinterpret_cast<const char *>(src);
          dst.size_ = sizeof(T);
        }

        template<>
        inline void putKey<tstring>(ROCKSDB_NAMESPACE::Slice &dst, const tstring *src) {
          dst.data_ = src->data();
          dst.size_ = src->size();
        }

        template<class T>
        inline void getValue(T *dst, const std::string &src, const size_t &n) {
          if (src.size() != n * sizeof(T)) {
            std::stringstream msg(std::stringstream::out);
            msg << "Expected " << n * sizeof(T)
                << " bytes, but " << src.size()
                << " bytes were returned by the database.";
            throw std::runtime_error(msg.str());
          }
          std::memcpy(dst, src.data(), src.size());
        }

        template<>
        inline void getValue<tstring>(tstring *dst, const std::string &src_, const size_t &n) {
          const char *src = src_.data();
          const char *const srcEnd = &src[src_.size()];
          const tstring *const dstEnd = &dst[n];

          for (; dst != dstEnd; ++dst) {
            const char *const srcSize = src;
            src += sizeof(STRING_SIZE_TYPE);
            if (src > srcEnd) {
              throw std::out_of_range("String value is malformed!");
            }
            const auto &size = *reinterpret_cast<const STRING_SIZE_TYPE *>(srcSize);

            const char *const srcData = src;
            src += size;
            if (src > srcEnd) {
              throw std::out_of_range("String value is malformed!");
            }
            dst->assign(srcData, size);
          }

          if (src != srcEnd) {
            throw std::runtime_error(
              "Database returned more values than the destination tensor could absorb."
            );
          }
        }

        template<class T>
        inline void putValue(ROCKSDB_NAMESPACE::PinnableSlice &dst, const T *src, const size_t &n) {
          dst.data_ = reinterpret_cast<const char *>(src);
          dst.size_ = sizeof(T) * n;
        }

        template<>
        inline void putValue<tstring>(
          ROCKSDB_NAMESPACE::PinnableSlice &dst_, const tstring *src, const size_t &n
        ) {
          std::string &dst = *dst_.GetSelf();
          dst.clear();

          // Concatenate the strings.
          const tstring *const srcEnd = &src[n];
          for (; src != srcEnd; ++src) {
            if (src->size() > std::numeric_limits<STRING_SIZE_TYPE>::max()) {
              throw std::runtime_error("String value is too large.");
            }
            const auto size = static_cast<STRING_SIZE_TYPE>(src->size());
            dst.append(reinterpret_cast<const char*>(&size), sizeof(size));
            dst.append(src->data(), size);
          }

          dst_.PinSelf();
        }

      }

      namespace _io {

        template<class T>
        inline void read(std::istream &src, T &dst) {
          if (!src.read(reinterpret_cast<char *>(&dst), sizeof(T))) {
            throw std::overflow_error("Unexpected end of file!");
          }
        }

        template<class T>
        inline T read(std::istream &src) { T tmp; read(src, tmp); return tmp; }

        template<class T>
        inline void write(std::ostream &dst, const T &src) {
          if (!dst.write(reinterpret_cast<const char *>(&src), sizeof(T))) {
            throw std::runtime_error("Writing file failed!");
          }
        }

        template<class T>
        inline void readKey(std::istream &src, std::string &dst) {
          dst.resize(sizeof(T));
          if (!src.read(&dst.front(), sizeof(T))) {
            throw std::overflow_error("Unexpected end of file!");
          }
        }

        template<>
        inline void readKey<tstring>(std::istream &src, std::string &dst) {
          const auto size = read<KEY_SIZE_TYPE>(src);
          dst.resize(size);
          if (!src.read(&dst.front(), size)) {
            throw std::overflow_error("Unexpected end of file!");
          }
        }

        template<class T>
        inline void writeKey(std::ostream &dst, const ROCKSDB_NAMESPACE::Slice &src) {
          write(dst, *reinterpret_cast<const T *>(src.data()));
        }

        template<>
        inline void writeKey<tstring>(std::ostream &dst, const ROCKSDB_NAMESPACE::Slice &src) {
          if (src.size() > std::numeric_limits<KEY_SIZE_TYPE>::max()) {
            throw std::overflow_error("String key is too long for RDB_KEY_SIZE_TYPE.");
          }
          const auto size = static_cast<KEY_SIZE_TYPE>(src.size());
          write(dst, size);
          if (!dst.write(src.data(), size)) {
            throw std::runtime_error("Writing file failed!");
          }
        }

        inline void readValue(std::istream &src, std::string &dst) {
          const auto size = read<VALUE_SIZE_TYPE>(src);
          dst.resize(size);
          if (!src.read(&dst.front(), size)) {
            throw std::overflow_error("Unexpected end of file!");
          }
        }

        inline void writeValue(std::ostream &dst, const ROCKSDB_NAMESPACE::Slice &src) {
          const auto size = static_cast<VALUE_SIZE_TYPE>(src.size());
          write(dst, &size);
          if (!dst.write(src.data(), size)) {
            throw std::runtime_error("Writing file failed!");
          }
        }

      }

      namespace _it {

        template<class T>
        inline void readKey(std::vector<T> &dst, const ROCKSDB_NAMESPACE::Slice &src) {
          if (src.size() != sizeof(T)) {
            std::stringstream msg(std::stringstream::out);
            msg << "Key size is out of bounds [ " << src.size() << " != " << sizeof(T) << " ].";
            throw std::out_of_range(msg.str());
          }
          dst.emplace_back(*reinterpret_cast<const T *>(src.data()));
        }

        template<>
        inline void readKey<tstring>(
          std::vector<tstring> &dst, const ROCKSDB_NAMESPACE::Slice &src
        ) {
          if (src.size() > std::numeric_limits<KEY_SIZE_TYPE>::max()) {
            std::stringstream msg(std::stringstream::out);
            msg << "Key size is out of bounds "
                << "[ " << src.size() << " > " << std::numeric_limits<KEY_SIZE_TYPE>::max() << "].";
            throw std::out_of_range(msg.str());
          }
          dst.emplace_back(src.data(), src.size());
        }

        template<class T>
        inline size_t readValue(std::vector<T> &dst, const ROCKSDB_NAMESPACE::Slice &src_) {
          const size_t n = src_.size() / sizeof(T);
          if (n * sizeof(T) != src_.size()) {
            std::stringstream msg(std::stringstream::out);
            msg << "Vector value is out of bounds "
                << "[ " << n * sizeof(T) << " != " << src_.size() << " ].";
            throw std::out_of_range(msg.str());
          }

          const T *const src = reinterpret_cast<const T *>(src_.data());
          dst.insert(dst.end(), src, &src[n]);
          return n;
        }

        template<>
        inline size_t readValue<tstring>(
          std::vector<tstring> &dst, const ROCKSDB_NAMESPACE::Slice &src_
        ) {
          const size_t dstSizePrev = dst.size();

          const char *src = src_.data();
          const char *const srcEnd = &src[src_.size()];

          while (src < srcEnd) {
            const char *const srcSize = src;
            src += sizeof(STRING_SIZE_TYPE);
            if (src > srcEnd) {
              throw std::out_of_range("String value is malformed!");
            }
            const auto &size = *reinterpret_cast<const STRING_SIZE_TYPE *>(srcSize);

            const char *const srcData = src;
            src += size;
            if (src > srcEnd) {
              throw std::out_of_range("String value is malformed!");
            }
            dst.emplace_back(srcData, size);
          }

          if (src != srcEnd) {
            throw std::out_of_range("String value is malformed!");
          }
          return dst.size() - dstSizePrev;
        }

      }

      class DBWrapper final {
      public:
        DBWrapper(const std::string &path, const bool &readOnly)
        : path_(path), readOnly_(readOnly), database_(nullptr) {
          ROCKSDB_NAMESPACE::Options options;
          options.create_if_missing = !readOnly;
          options.manual_wal_flush = false;

          // Create or connect to the RocksDB database.
          std::vector<std::string> colFamilies;
          #if __cplusplus >= 201703L
          if (!std::filesystem::exists(path)) {
            colFamilies.push_back(ROCKSDB_NAMESPACE::kDefaultColumnFamilyName);
          }
          else if (std::filesystem::is_directory(path)){
            ROCKSDB_OK(ROCKSDB_NAMESPACE::DB::ListColumnFamilies(options, path, &colFamilies));
          }
          else {
            throw std::runtime_error("Provided database path is invalid.");
          }
          #else
          struct stat dbPathStat{};
          if (stat(path.c_str(), &dbPathStat) == 0) {
            if (S_ISDIR(dbPathStat.st_mode)) {
              ROCKSDB_OK(ROCKSDB_NAMESPACE::DB::ListColumnFamilies(options, path, &colFamilies));
            }
            else {
              throw std::runtime_error("Provided database path is invalid.");
            }
          }
          else {
            colFamilies.push_back(ROCKSDB_NAMESPACE::kDefaultColumnFamilyName);
          }
          #endif

          ROCKSDB_NAMESPACE::ColumnFamilyOptions colFamilyOptions;
          std::vector<ROCKSDB_NAMESPACE::ColumnFamilyDescriptor> colDescriptors;
          for (const auto &cf : colFamilies) {
            colDescriptors.emplace_back(cf, colFamilyOptions);
          }

          ROCKSDB_NAMESPACE::DB *db;
          std::vector<ROCKSDB_NAMESPACE::ColumnFamilyHandle *> chs;
          if (readOnly) {
            ROCKSDB_OK(ROCKSDB_NAMESPACE::DB::OpenForReadOnly(
              options, path, colDescriptors, &chs, &db
            ));
          }
          else {
            ROCKSDB_OK(ROCKSDB_NAMESPACE::DB::Open(
              options, path, colDescriptors, &chs, &db
            ));
          }
          database_.reset(db);

          // Maintain map of the available column handles for quick access.
          for (const auto &colHandle : chs) {
            colHandles[colHandle->GetName()] = colHandle;
          }

          LOG(INFO) << "Connected to database \'" << path_ << "\'.";
        }

        ~DBWrapper() {
          for (const auto &ch : colHandles) {
            if (!readOnly_) {
              database_->FlushWAL(true);
            }
            database_->DestroyColumnFamilyHandle(ch.second);
          }
          colHandles.clear();
          database_.reset();
          LOG(INFO) << "Disconnected from database \'" << path_ << "\'.";
        }

        inline ROCKSDB_NAMESPACE::DB *database() { return database_.get(); }

        inline const std::string &path() const { return path_; }

        inline bool readOnly() const { return readOnly_; }

        void deleteColumn(const std::string &colName) {
          mutex_lock guard(lock);

          // Try to locate column handle, and return if it anyway doe not exist.
          const auto &item = colHandles.find(colName);
          if (item == colHandles.end()) {
            return;
          }

          // If a modification would be required make sure we are not in readonly mode.
          if (readOnly_) {
            throw std::runtime_error("Cannot delete a column in readonly mode.");
          }

          // Perform actual removal.
          ROCKSDB_OK(database_->DropColumnFamily(item->second));
          ROCKSDB_OK(database_->DestroyColumnFamilyHandle(item->second));
          colHandles.erase(colName);
        }

        template<class T>
        T withColumn(
          const std::string &colName,
          std::function<T(ROCKSDB_NAMESPACE::ColumnFamilyHandle *const)> fn
        ) {
          tf_shared_lock guard(lock);

          // Invoke the function while we are guarded.
          const auto &colHandle = getColumn(colName);
          return fn(colHandle);
        }

        inline ROCKSDB_NAMESPACE::DB *operator->() { return database_.get(); }

      private:
        ROCKSDB_NAMESPACE::ColumnFamilyHandle *getColumn(const std::string &colName) {
          // Try to locate column handle.
          const auto &item = colHandles.find(colName);
          if (item != colHandles.end()) {
            return item->second;
          }

          // Do not create an actual column handle in readonly mode.
          if (readOnly_) {
            return nullptr;
          }

          // Create a new column handle.
          ROCKSDB_NAMESPACE::ColumnFamilyOptions colFamilyOptions;
          ROCKSDB_NAMESPACE::ColumnFamilyHandle *colHandle;
          ROCKSDB_OK(database_->CreateColumnFamily(colFamilyOptions, colName, &colHandle));
          colHandles[colName] = colHandle;

          return colHandle;
        }

      private:
        const std::string path_;
        const bool readOnly_;
        std::unique_ptr<ROCKSDB_NAMESPACE::DB> database_;

        mutex lock;
        std::unordered_map<std::string, ROCKSDB_NAMESPACE::ColumnFamilyHandle*> colHandles;
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
        std::shared_ptr<DBWrapper> connect(
          const std::string &databasePath, const bool &readOnly
        ) {
          mutex_lock guard(lock);

          // Try to find database, or open it if it is not open yet.
          std::shared_ptr<DBWrapper> db;
          auto pos = wrappers.find(databasePath);
          if (pos != wrappers.end()) {
            db = pos->second.lock();
          }
          else {
            db.reset(new DBWrapper(databasePath, readOnly), deleter);
            wrappers[databasePath] = db;
          }

          // Suicide, if the desired access level is below the available access level.
          if (readOnly < db->readOnly()) {
            throw std::runtime_error("Cannot simultaneously open database in read + write mode.");
          }

          return db;
        }

      private:
        static void deleter(DBWrapper *wrapper) {
          static std::default_delete<DBWrapper> defaultDeleter;

          DBWrapperRegistry &registry = instance();
          const std::string path = wrapper->path();

          // Make sure we are alone.
          mutex_lock guard(registry.lock);

          // Destroy the wrapper.
          defaultDeleter(wrapper);
          // LOG(INFO) << "Database wrapper " << path << " has been deleted.";

          // Locate the corresponding weak_ptr and evict it.
          auto pos = registry.wrappers.find(path);
          if (pos == registry.wrappers.end()) {
            LOG(ERROR) << "Unknown database wrapper. How?";
          }
          else if (pos->second.expired()) {
            registry.wrappers.erase(pos);
            // LOG(INFO) << "Database wrapper " << path << " evicted.";
          }
          else {
            LOG(ERROR) << "Registry is in an inconsistent state. This is very bad...";
          }
        }

      private:
        mutex lock;
        std::unordered_map<std::string, std::weak_ptr<DBWrapper>> wrappers;
      };

      template<class K, class V>
      class RocksDBTableOfTensors final : public PersistentStorageLookupInterface {
      public:
        /* --- BASE INTERFACE ------------------------------------------------------------------- */
        RocksDBTableOfTensors(OpKernelContext *ctx, OpKernel *kernel)
        : readOnly(false), estimateSize(false), dirtyCount(0) {
          OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "value_shape", &valueShape));
          OP_REQUIRES(ctx, TensorShapeUtils::IsVector(valueShape), errors::InvalidArgument(
            "Default value must be a vector, got shape ", valueShape.DebugString()
          ));

          OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "database_path", &databasePath));
          OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "embedding_name", &embeddingName));
          OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "read_only", &readOnly));
          OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "estimate_size", &estimateSize));
          flushInterval = 1;
          OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "export_path", &defaultExportPath));

          db = DBWrapperRegistry::instance().connect(databasePath, readOnly);
          LOG(INFO) << "Acquired reference to database wrapper " << db->path()
                    << " [ #refs = " << db.use_count() << " ].";
        }

        ~RocksDBTableOfTensors() override {
          LOG(INFO) << "Dropping reference to database wrapper " << db->path()
                    << " [ #refs = " << db.use_count() << " ].";
        }

        DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }
        TensorShape key_shape() const override { return TensorShape(); }

        DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }
        TensorShape value_shape() const override { return valueShape; }

        size_t size() const override {
          auto fn = [this](
            ROCKSDB_NAMESPACE::ColumnFamilyHandle *const colHandle
          ) -> size_t {
            // If allowed, try to just estimate of the number of keys.
            if (estimateSize) {
              uint64_t numKeys;
              if ((*db)->GetIntProperty(
                colHandle, ROCKSDB_NAMESPACE::DB::Properties::kEstimateNumKeys, &numKeys
              )) {
                return numKeys;
              }
            }

            // Alternative method, walk the entire database column and count the keys.
            std::unique_ptr<ROCKSDB_NAMESPACE::Iterator> iter(
              (*db)->NewIterator(readOptions, colHandle)
            );
            iter->SeekToFirst();

            size_t numKeys = 0;
            for (; iter->Valid(); iter->Next()) { ++numKeys; }
            return numKeys;
          };

          return db->withColumn<size_t>(embeddingName, fn);
        }

      public:
        /* --- LOOKUP --------------------------------------------------------------------------- */
        Status Clear(OpKernelContext *ctx) override {
          db->deleteColumn(embeddingName);
          return Status::OK();
        }

        Status Find(
          OpKernelContext *ctx, const Tensor &keys, Tensor *values, const Tensor &default_value
        ) override {
          if (
            keys.dtype() != key_dtype() ||
            values->dtype() != value_dtype() ||
            default_value.dtype() != value_dtype()
          ) {
            return errors::InvalidArgument("Tensor dtypes are incompatible!");
          }

          const size_t numKeys = keys.dim_size(0);
          const size_t numValues = values->dim_size(0);
          if (numKeys != numValues) {
            return errors::InvalidArgument(
              "First dimension of the key and value tensors does not match!"
            );
          }
          const size_t valuesPerDim0 = values->NumElements() / numValues;

          const K *k = static_cast<K *>(keys.data());
          V *const v = static_cast<V *>(values->data());

          const V *const d = static_cast<V *>(default_value.data());
          const size_t dSize = default_value.NumElements();

          if (dSize % valuesPerDim0 != 0) {
            return errors::InvalidArgument(
              "The shapes of the values and default_value tensors are not compatible."
            );
          }

          auto fn = [this, &numKeys, &valuesPerDim0, &k, &v, &d, &dSize](
            ROCKSDB_NAMESPACE::ColumnFamilyHandle *const colHandle
          ) -> Status {
            size_t vOffset = 0;

            if (numKeys < BATCH_SIZE_MIN) {
              const K *const kEnd = &k[numKeys];
              ROCKSDB_NAMESPACE::Slice kSlice;

              for (; k != kEnd; ++k, vOffset += valuesPerDim0) {
                _if::putKey(kSlice, k);
                std::string vSlice;

                auto status = colHandle
                  ? (*db)->Get(readOptions, colHandle, kSlice, &vSlice)
                  : ROCKSDB_NAMESPACE::Status::NotFound();

                if (status.ok()) {
                  _if::getValue(&v[vOffset], vSlice, valuesPerDim0);
                }
                else if (status.IsNotFound()) {
                  std::copy_n(&d[vOffset % dSize], valuesPerDim0, &v[vOffset]);
                }
                else {
                  throw std::runtime_error(status.getState());
                }
              }
            }
            else {
              // There is no point in filling this vector every time as long as it is big enough.
              if (!colHandleCache.empty() && colHandleCache.front() != colHandle) {
                std::fill(colHandleCache.begin(), colHandleCache.end(), colHandle);
              }
              if (colHandleCache.size() < numKeys) {
                colHandleCache.insert(
                  colHandleCache.end(), numKeys - colHandleCache.size(), colHandle
                );
              }

              // Query all keys using a single Multi-Get.
              std::vector<std::string> vSlices;
              std::vector<ROCKSDB_NAMESPACE::Slice> kSlices(numKeys);
              for (size_t i = 0; i < numKeys; ++i) {
                _if::putKey(kSlices[i], &k[i]);
              }

              const std::vector<ROCKSDB_NAMESPACE::Status> &statuses = colHandle
                ? (*db)->MultiGet(readOptions, colHandleCache, kSlices, &vSlices)
                : std::vector<ROCKSDB_NAMESPACE::Status>(
                  numKeys, ROCKSDB_NAMESPACE::Status::NotFound()
                );

              if (statuses.size() != numKeys) {
                std::stringstream msg(std::stringstream::out);
                msg << "Requested " << numKeys
                    << " keys, but only got " << statuses.size()
                    << " responses.";
                throw std::runtime_error(msg.str());
              }

              // Process results.
              for (size_t i = 0; i < numKeys; ++i, vOffset += valuesPerDim0) {
                const auto &status = statuses[i];
                const auto &vSlice = vSlices[i];

                if (status.ok()) {
                  _if::getValue(&v[vOffset], vSlice, valuesPerDim0);
                }
                else if (status.IsNotFound()) {
                  std::copy_n(&d[vOffset % dSize], valuesPerDim0, &v[vOffset]);
                }
                else {
                  throw std::runtime_error(status.getState());
                }
              }
            }

            return Status::OK();
          };

          return db->withColumn<Status>(embeddingName, fn);
        }

        Status Insert(OpKernelContext *ctx, const Tensor &keys, const Tensor &values) override {
          if (keys.dtype() != key_dtype() || values.dtype() != value_dtype()) {
            return errors::InvalidArgument("Tensor dtypes are incompatible!");
          }

          const size_t numKeys = keys.dim_size(0);
          const size_t numValues = values.dim_size(0);
          if (numKeys != numValues) {
            return errors::InvalidArgument(
              "First dimension of the key and value tensors does not match!"
            );
          }
          const size_t valuesPerDim0 = values.NumElements() / numValues;

          const K *k = static_cast<K *>(keys.data());
          const V *v = static_cast<V *>(values.data());

          auto fn = [this, &numKeys, &valuesPerDim0, &k, &v](
            ROCKSDB_NAMESPACE::ColumnFamilyHandle *const colHandle
          ) -> Status {
            if (readOnly || !colHandle) {
              return errors::PermissionDenied("Cannot insert in read_only mode.");
            }

            const K *const kEnd = &k[numKeys];
            ROCKSDB_NAMESPACE::Slice kSlice;
            ROCKSDB_NAMESPACE::PinnableSlice vSlice;

            if (numKeys < BATCH_SIZE_MIN) {
              for (; k != kEnd; ++k, v += valuesPerDim0) {
                _if::putKey(kSlice, k);
                _if::putValue(vSlice, v, valuesPerDim0);
                ROCKSDB_OK((*db)->Put(writeOptions, colHandle, kSlice, vSlice));
              }
            }
            else {
              ROCKSDB_NAMESPACE::WriteBatch batch;
              for (; k != kEnd; ++k, v += valuesPerDim0) {
                _if::putKey(kSlice, k);
                _if::putValue(vSlice, v, valuesPerDim0);
                ROCKSDB_OK(batch.Put(colHandle, kSlice, vSlice));
              }
              ROCKSDB_OK((*db)->Write(writeOptions, &batch));
            }

            // Handle interval flushing.
            dirtyCount += 1;
            if (dirtyCount % flushInterval == 0) {
              ROCKSDB_OK((*db)->FlushWAL(true));
            }

            return Status::OK();
          };

          return db->withColumn<Status>(embeddingName, fn);
        }

        Status Remove(OpKernelContext *ctx, const Tensor &keys) override {
          if (keys.dtype() != key_dtype()) {
            return errors::InvalidArgument("Tensor dtypes are incompatible!");
          }

          const size_t numKeys = keys.dim_size(0);
          const K *k = static_cast<K *>(keys.data());

          auto fn = [this, &numKeys, &k](
            ROCKSDB_NAMESPACE::ColumnFamilyHandle *const colHandle
          ) -> Status {
            if (readOnly || !colHandle) {
              return errors::PermissionDenied("Cannot remove in read_only mode.");
            }

            const K *const kEnd = &k[numKeys];
            ROCKSDB_NAMESPACE::Slice kSlice;

            if (numKeys < BATCH_SIZE_MIN) {
              for (; k != kEnd; ++k) {
                _if::putKey(kSlice, k);
                ROCKSDB_OK((*db)->Delete(writeOptions, colHandle, kSlice));
              }
            }
            else {
              ROCKSDB_NAMESPACE::WriteBatch batch;
              for (; k != kEnd; ++k) {
                _if::putKey(kSlice, k);
                ROCKSDB_OK(batch.Delete(colHandle, kSlice));
              }
              ROCKSDB_OK((*db)->Write(writeOptions, &batch));
            }

            // Handle interval flushing.
            dirtyCount += 1;
            if (dirtyCount % flushInterval == 0) {
              ROCKSDB_OK((*db)->FlushWAL(true));
            }

            return Status::OK();
          };

          return db->withColumn<Status>(embeddingName, fn);
        }

        /* --- IMPORT / EXPORT ------------------------------------------------------------------ */
        Status ExportValues(OpKernelContext *ctx) override {
          if (defaultExportPath.empty()) {
            return ExportValuesToTensor(ctx);
          }
          else {
            return ExportValuesToFile(ctx, defaultExportPath);
          }
        }
        Status ImportValues(
          OpKernelContext *ctx, const Tensor &keys, const Tensor &values
        ) override {
          if (defaultExportPath.empty()) {
            return ImportValuesFromTensor(ctx, keys, values);
          }
          else {
            return ImportValuesFromFile(ctx, defaultExportPath);
          }
        }

        Status ExportValuesToFile(OpKernelContext *ctx, const std::string &path) {
          std::ofstream file(path, std::ofstream::binary);
          if (!file) {
            return errors::Unknown("Could not open dump file.");
          }

          // Create file header.
          _io::write(file, FILE_MAGIC);
          _io::write(file, FILE_VERSION);
          _io::write(file, key_dtype());
          _io::write(file, value_dtype());

          auto fn = [this, &file](
            ROCKSDB_NAMESPACE::ColumnFamilyHandle *const colHandle
          ) -> Status {
            // Iterate through entries one-by-one and append them to the file.
            std::unique_ptr<ROCKSDB_NAMESPACE::Iterator> iter(
              (*db)->NewIterator(readOptions, colHandle)
            );
            iter->SeekToFirst();

            for (; iter->Valid(); iter->Next()) {
              _io::writeKey<K>(file, iter->key());
              _io::writeValue(file, iter->value());
            }

            return Status::OK();
          };

          return db->withColumn<Status>(embeddingName, fn);
        }
        Status ImportValuesFromFile(OpKernelContext *ctx, const std::string &path) {
          // Make sure the column family is clean.
          const auto &clearStatus = Clear(ctx);
          if (!clearStatus.ok()) {
            return clearStatus;
          }

          std::ifstream file(path, std::ifstream::binary);
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
            return errors::Unimplemented("File version ", version, " is not supported");
          }
          const auto kDType = _io::read<DataType>(file);
          const auto vDType = _io::read<DataType>(file);
          if (kDType != key_dtype() || vDType != value_dtype()) {
            return errors::Internal(
              "DataType of file [k=", kDType, ", v=", vDType, "] ",
              "do not match module DataType [k=", key_dtype(), ", v=", value_dtype(), "]."
            );
          }

          auto fn = [this, &file](
            ROCKSDB_NAMESPACE::ColumnFamilyHandle *const colHandle
          ) -> Status {
            if (readOnly || !colHandle) {
              return errors::PermissionDenied("Cannot import in read_only mode.");
            }

            // Read payload and subsequently populate column family.
            ROCKSDB_NAMESPACE::WriteBatch batch;

            ROCKSDB_NAMESPACE::PinnableSlice kSlice;
            ROCKSDB_NAMESPACE::PinnableSlice vSlice;

            while (!file.eof()) {
              _io::readKey<K>(file, *kSlice.GetSelf()); kSlice.PinSelf();
              _io::readValue(file, *vSlice.GetSelf()); vSlice.PinSelf();
              ROCKSDB_OK(batch.Put(colHandle, kSlice, vSlice));

              // If batch reached target size, write to database.
              if (batch.Count() >= BATCH_SIZE_MAX) {
                ROCKSDB_OK((*db)->Write(writeOptions, &batch));
                batch.Clear();
              }
            }

            // Write remaining entries, if any.
            if (batch.Count()) {
              ROCKSDB_OK((*db)->Write(writeOptions, &batch));
            }

            // Handle interval flushing.
            dirtyCount += 1;
            if (dirtyCount % flushInterval == 0) {
              ROCKSDB_OK((*db)->FlushWAL(true));
            }

            return Status::OK();
          };

          return db->withColumn<Status>(embeddingName, fn);
        }

        Status ExportValuesToTensor(OpKernelContext *ctx) {
          // Fetch data from database.
          std::vector<K> kBuffer;
          std::vector<V> vBuffer;
          int64 valueCount = -1;

          auto fn = [this, &kBuffer, &vBuffer, &valueCount](
            ROCKSDB_NAMESPACE::ColumnFamilyHandle *const colHandle
          ) -> Status {
            std::unique_ptr<ROCKSDB_NAMESPACE::Iterator> iter(
              (*db)->NewIterator(readOptions, colHandle)
            );
            iter->SeekToFirst();

            for (; iter->Valid(); iter->Next()) {
              const auto &kSlice = iter->key();
              _it::readKey(kBuffer, kSlice);

              const auto vSlice = iter->value();
              const int64 vSize = _it::readValue(vBuffer, vSlice);

              // Make sure we have a square tensor.
              if (valueCount < 0) {
                valueCount = vSize;
              }
              else if (vSize != valueCount) {
                return errors::Internal("The returned tensor sizes differ.");
              }
            }

            return Status::OK();
          };

          const auto &status = db->withColumn<Status>(embeddingName, fn);
          if (!status.ok()) {
            return status;
          }
          valueCount = std::max(valueCount, 0LL);
          const auto numKeys = static_cast<int64>(kBuffer.size());

          // Populate keys tensor.
          Tensor *kTensor;
          TF_RETURN_IF_ERROR(ctx->allocate_output(
            "keys", TensorShape({numKeys}), &kTensor
          ));
          K *const k = reinterpret_cast<K *>(kTensor->data());
          std::copy(kBuffer.begin(), kBuffer.end(), k);

          // Populate values tensor.
          Tensor *vTensor;
          TF_RETURN_IF_ERROR(ctx->allocate_output(
            "values", TensorShape({numKeys, valueCount}), &vTensor
          ));
          V *const v = reinterpret_cast<V *>(vTensor->data());
          std::copy(vBuffer.begin(), vBuffer.end(), v);

          return status;
        }
        Status ImportValuesFromTensor(
          OpKernelContext *ctx, const Tensor &keys, const Tensor &values
        ) {
          // Make sure the column family is clean.
          const auto &clearStatus = Clear(ctx);
          if (!clearStatus.ok()) {
            return clearStatus;
          }

          // Just call normal insertion function.
          return Insert(ctx, keys, values);
        }

      protected:
        TensorShape valueShape;
        std::string databasePath;
        std::string embeddingName;
        bool readOnly;
        bool estimateSize;
        size_t flushInterval;
        std::string defaultExportPath;

        std::shared_ptr<DBWrapper> db;
        ROCKSDB_NAMESPACE::ReadOptions readOptions;
        ROCKSDB_NAMESPACE::WriteOptions writeOptions;
        size_t dirtyCount;

        std::vector<ROCKSDB_NAMESPACE::ColumnFamilyHandle *> colHandleCache;
      };

      #undef ROCKSDB_OK

      /* --- KERNEL REGISTRATION ---------------------------------------------------------------- */
      #define ROCKSDB_REGISTER_KERNEL_BUILDER(key_dtype, value_dtype)                                \
        REGISTER_KERNEL_BUILDER(                                                                 \
          Name(PREFIX_OP_NAME(RocksdbTableOfTensors))                                            \
            .Device(DEVICE_CPU)                                                                  \
            .TypeConstraint<key_dtype>("key_dtype")                                              \
            .TypeConstraint<value_dtype>("value_dtype"),                                         \
          RocksDBTableOp<RocksDBTableOfTensors<key_dtype, value_dtype>, key_dtype, value_dtype>  \
        )

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
    }  // namespace rocksdb_lookup

    /* --- OP KERNELS --------------------------------------------------------------------------- */
    class RocksDBTableOpKernel : public OpKernel {
    public:
      explicit RocksDBTableOpKernel(OpKernelConstruction *ctx)
      : OpKernel(ctx)
      , expected_input_0_(ctx->input_type(0) == DT_RESOURCE ? DT_RESOURCE : DT_STRING_REF) {
      }

    protected:
      Status LookupResource(
        OpKernelContext *ctx, const ResourceHandle &p, LookupInterface **value
      ) {
        return ctx->resource_manager()->Lookup<LookupInterface, false>(
          p.container(), p.name(), value
        );
      }

      Status GetResourceHashTable(
        StringPiece input_name, OpKernelContext *ctx, LookupInterface **table
      ) {
        const Tensor *handle_tensor;
        TF_RETURN_IF_ERROR(ctx->input(input_name, &handle_tensor));
        const auto &handle = handle_tensor->scalar<ResourceHandle>()();
        return LookupResource(ctx, handle, table);
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

        auto *rocksTable = dynamic_cast<PersistentStorageLookupInterface *>(table);

        int64 memory_used_before = 0;
        if (ctx->track_allocations()) {
          memory_used_before = table->MemoryUsed();
        }
        OP_REQUIRES_OK(ctx, rocksTable->Clear(ctx));
        if (ctx->track_allocations()) {
          ctx->record_persistent_memory_allocation(table->MemoryUsed() - memory_used_before);
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

        DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(), table->value_dtype()};
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

        DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(), table->value_dtype()};
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
          ctx->record_persistent_memory_allocation(table->MemoryUsed() - memory_used_before);
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

        DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(), table->value_dtype()};
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
          ctx->record_persistent_memory_allocation(table->MemoryUsed() - memory_used_before);
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
          ctx->record_persistent_memory_allocation(table->MemoryUsed() - memory_used_before);
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
      Name(PREFIX_OP_NAME(RocksdbTableClear)).Device(DEVICE_CPU), RocksDBTableClear
    );
    REGISTER_KERNEL_BUILDER(
      Name(PREFIX_OP_NAME(RocksdbTableExport)).Device(DEVICE_CPU), RocksDBTableExport
    );
    REGISTER_KERNEL_BUILDER(
      Name(PREFIX_OP_NAME(RocksdbTableFind)).Device(DEVICE_CPU), RocksDBTableFind
    );
    REGISTER_KERNEL_BUILDER(
      Name(PREFIX_OP_NAME(RocksdbTableImport)).Device(DEVICE_CPU), RocksDBTableImport
    );
    REGISTER_KERNEL_BUILDER(
      Name(PREFIX_OP_NAME(RocksdbTableInsert)).Device(DEVICE_CPU), RocksDBTableInsert
    );
    REGISTER_KERNEL_BUILDER(
      Name(PREFIX_OP_NAME(RocksdbTableRemove)).Device(DEVICE_CPU), RocksDBTableRemove
    );
    REGISTER_KERNEL_BUILDER(
      Name(PREFIX_OP_NAME(RocksdbTableSize)).Device(DEVICE_CPU), RocksDBTableSize
    );

  }  // namespace recommenders_addons
}  // namespace tensorflow
