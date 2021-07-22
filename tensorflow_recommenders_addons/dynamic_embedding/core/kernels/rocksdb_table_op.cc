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

#include <ios>
#include <iostream>
#include <iomanip>
#include <fstream>
#if __cplusplus >= 201703L
  #include <filesystem>
#else
  #include <sys/stat.h>
#endif
#include <sstream>
#include "../utils/utils.h"
#include "rocksdb_table_op.h"
#include "rocksdb/db.h"

namespace tensorflow {
  namespace recommenders_addons {
    namespace lookup {

      static const int64 RDB_BATCH_MODE_MIN_QUERY_SIZE = 2;
      static const uint32_t RDB_BATCH_MODE_MAX_QUERY_SIZE = 128;
      static const uint32_t RDB_EXPORT_FILE_MAGIC = ( // TODO: Little endian / big endian conversion?
        (static_cast<uint32_t>('T') << 0) |
        (static_cast<uint32_t>('F') << 8) |
        (static_cast<uint32_t>('K') << 16) |
        (static_cast<uint32_t>('V') << 24)
      );
      static const uint32_t RDB_EXPORT_FILE_VERSION = 1;
      static const char RDB_EXPORT_PATH[] = "/tmp/db.dump";
      static const bool RDB_VERBOSITY = 100;


      #define RDB_OK(EXPR)                                    \
         do {                                                 \
            const rocksdb::Status s = EXPR;                   \
            if (!s.ok()) {                                    \
              std::stringstream msg(std::stringstream::out);  \
              msg << "RocksDB error " << s.code()             \
                  << "; reason: " << s.getState()             \
                  << "; expr: " << #EXPR;                     \
              throw std::runtime_error(msg.str());            \
            }                                                 \
         } while (0)

      template<class T>
      inline void copyToTensor(T *dst, const std::string &slice, const int64 &numValues) {
        if (slice.size() != numValues * sizeof(T)) {
          std::stringstream msg;
          msg << "Expected " << numValues * sizeof(T)
              << " bytes, but " << slice.size()
              << " bytes were returned by RocksDB.";
          throw std::runtime_error(msg.str());
        }
        std::memcpy(dst, slice.data(), slice.size());
      }

      template<>
      inline void copyToTensor<tstring>(
        tstring *dst, const std::string &slice, const int64 &numValues
      ) {
        const char *src = slice.data();
        const char *const srcEnd = &src[slice.size()];
        const tstring *const dstEnd = &dst[numValues];

        for (; dst != dstEnd; ++dst) {
          if (src + sizeof(uint32_t) > srcEnd) {
            throw std::runtime_error(
              "Something is very..very..very wrong. Buffer overflow immanent!"
            );
          }
          const uint32_t length = *reinterpret_cast<const uint32_t *>(src);
          src += sizeof(uint32_t);

          if (src + length > srcEnd) {
            throw std::runtime_error(
              "Something is very..very..very wrong. Buffer overflow immanent!"
            );
          }
          dst->assign(src, length);
          src += length;
        }

        if (src != srcEnd) {
          throw std::runtime_error(
            "RocksDB returned more values than the destination tensor could absorb."
          );
        }
      }

      template<class T>
      inline void assignSlice(rocksdb::Slice &dst, const T *src) {
        dst.data_ = reinterpret_cast<const char *>(src);
        dst.size_ = sizeof(T);
      }

      template<>
      inline void assignSlice<tstring>(rocksdb::Slice &dst, const tstring *src) {
        dst.data_ = src->data();
        dst.size_ = src->size();
      }

      template<class T>
      inline void assignSlice(rocksdb::PinnableSlice &dst, const T *src, const int64 numValues) {
        dst.data_ = reinterpret_cast<const char *>(src);
        dst.size_ = numValues * sizeof(T);
      }

      template<>
      void assignSlice<tstring>(
        rocksdb::PinnableSlice &dst, const tstring *src, const int64 numValues
      ) {
        // Allocate memory to be returned.
        std::string* d = dst.GetSelf();
        d->clear();

        // Concatenate the strings.
        const tstring *const srcEnd = &src[numValues];
        for (; src != srcEnd; ++src) {
          if (src->size() > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Value size is too large.");
          }
          uint32_t size = src->size();
          d->append(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
          d->append(*src);
        }
        dst.PinSelf();
      }

      class RocksDBLink {
      public:
        RocksDBLink(const std::string &path, const bool &readOnly)
        : db(nullptr), readOnly_(readOnly) {
          rocksdb::Options options;
          options.create_if_missing = !readOnly;
          options.manual_wal_flush = true;

          // Create or connect to the RocksDB database.
          std::vector<std::string> colFamilies;
          #if __cplusplus >= 201703L
          if (!std::filesystem::exists(dbPath)) {
            colFamilies.push_back(ROCKSDB_NAMESPACE::kDefaultColumnFamilyName);
          }
          else if (std::filesystem::is_directory(dbPath)){
            RDB_OK(rocksdb::DB::ListColumnFamilies(options, dbPath, &colFamilies));
          }
          else {
            throw std::runtime_error("Provided database path is invalid.");
          }
          #else
          struct stat dbPathStat;
          if (stat(path.c_str(), &dbPathStat) == 0) {
            if (S_ISDIR(dbPathStat.st_mode)) {
              RDB_OK(rocksdb::DB::ListColumnFamilies(options, path, &colFamilies));
            }
            else {
              throw std::runtime_error("Provided database path is invalid.");
            }
          }
          else {
            colFamilies.push_back(ROCKSDB_NAMESPACE::kDefaultColumnFamilyName);
          }
          #endif

          std::vector<rocksdb::ColumnFamilyDescriptor> colDescriptors;
          for (const auto &cf : colFamilies) {
            colDescriptors.emplace_back(cf, rocksdb::ColumnFamilyOptions());
          }

          std::vector<rocksdb::ColumnFamilyHandle *> chs;
          if (readOnly) {
            RDB_OK(rocksdb::DB::OpenForReadOnly(options, path, colDescriptors, &chs, &db));
          }
          else {
            RDB_OK(rocksdb::DB::Open(options, path, colDescriptors, &chs, &db));
          }

          // Maintain map of the available column handles for quick access.
          for (const auto &colHandle : chs) {
            colHandles[colHandle->GetName()] = colHandle;
          }
        }

        ~RocksDBLink() {
          for (const auto &item : colHandles) {
            if (!readOnly_) {
              db->FlushWAL(true);
            }
            db->DestroyColumnFamilyHandle(item.second);
          }
          colHandles.clear();
          delete db;
          db = nullptr;
        }

        rocksdb::ColumnFamilyHandle *getColumn(const std::string &colName) {
          // Make sure we are alone.
          std::lock_guard<std::mutex> guard(lock);

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
          rocksdb::ColumnFamilyHandle *colHandle;
          RDB_OK(db->CreateColumnFamily(rocksdb::ColumnFamilyOptions(), colName, &colHandle));
          colHandles[colName] = colHandle;
          return colHandle;
        }

        Status deleteColumn(const std::string &colName) {
          // Make sure we are alone.
          std::lock_guard<std::mutex> guard(lock);

          // Try to locate column handle, and return if it anyway doe not exist.
          const auto &item = colHandles.find(colName);
          if (item == colHandles.end()) {
            return Status::OK();
          }

          // If a modification would be required make sure we are not in readonly mode.
          if (readOnly_) {
            return errors::PermissionDenied("Cannot delete a column in readonly mode.");
          }

          // Perform actual removal.
          RDB_OK(db->DropColumnFamily(item->second));
          RDB_OK(db->DestroyColumnFamilyHandle(item->second));
          colHandles.erase(colName);
          return Status::OK();
        }

        bool readOnly() const { return readOnly_; }

        rocksdb::DB *operator->() { return db; }

      private:
        rocksdb::DB *db;
        bool readOnly_;
        mutable std::mutex lock;
        std::unordered_map<std::string, rocksdb::ColumnFamilyHandle*> colHandles;
      };

      class RocksDBConnectionPool {
      public:
        static RocksDBConnectionPool &instance() {
          static auto instance = new RocksDBConnectionPool();
          return *instance;
        }

      public:
        RocksDBConnectionPool() = default;

        ~RocksDBConnectionPool() {
          databases.clear();
        }

        RocksDBLink *open(const std::string &path, const bool &readOnly) {
          // Make sure we are alone.
          std::lock_guard<std::mutex> guard(lock);

          // Try to find database and open it if it is not open yet.
          auto links = databases.find(path);
          if (links == databases.end()) {
            databases[path] = {std::make_unique<RocksDBLink>(path, readOnly), 1};
            links = databases.find(path);
          }

          auto &link = links->second;
          RocksDBLink *db = std::get<0>(link).get();
          if (readOnly < db->readOnly()) {
            throw std::runtime_error("Cannot simultaneously open database in read + write mode.");
          }
          std::get<1>(link) += 1;
          return db;
        }

        void close(const std::string &path) {
          auto links = databases.find(path);
          if (links == databases.end()) {
            throw std::runtime_error("Unknown database.");
          }
          auto &link = links->second;
          std::get<1>(link) -= 1;
          if (std::get<1>(link) == 0) {
            databases.erase(path);
          }
        }

      private:
        std::mutex lock;
        std::unordered_map<std::string, std::tuple<std::unique_ptr<RocksDBLink>, long>> databases;
      };

      template<class K, class V>
      class RocksDBTableOfTensors final : public ClearableLookupInterface {
      public:
        /* --- BASE INTERFACE ------------------------------------------------------------------- */
        RocksDBTableOfTensors(OpKernelContext *ctx, OpKernel *kernel)
        : readOnly(false), estimateSize(true), flushInterval(1)
        , writeOpCount(0), prevColHandle(nullptr) {
          OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "value_shape", &valueShape));
          OP_REQUIRES(ctx, TensorShapeUtils::IsVector(valueShape), errors::InvalidArgument(
            "Default value must be a vector, got shape ", valueShape.DebugString()
          ));

          OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "database_path", &path));
          OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "embedding_name", &embeddingName));
          OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "read_only", &readOnly));
          OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "estimate_size", &estimateSize));

          // Open the database.
          link = RocksDBConnectionPool::instance().open(path, readOnly);
        }

        ~RocksDBTableOfTensors() override {
          colHandleCache.clear();
          link = nullptr;
          RocksDBConnectionPool::instance().close(path);
        }

        DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }

        DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }

        TensorShape key_shape() const override { return TensorShape(); }

        size_t size() const override {
          rocksdb::ColumnFamilyHandle *const colHandle = GetColumnHandle();

          // If allowed, try to just estimate of the number of keys.
          if (estimateSize) {
            uint64_t numKeys;
            if ((*link)->GetIntProperty(
              colHandle, rocksdb::DB::Properties::kEstimateNumKeys, &numKeys
            )) {
              return numKeys;
            }
          }

          // Alternative method, walk the entire database column and count the keys.
          std::unique_ptr<rocksdb::Iterator> iter((*link)->NewIterator(readOptions, colHandle));
          iter->SeekToFirst();

          size_t numKeys = 0;
          for (; iter->Valid(); iter->Next()) {
            // std::cout << "[ "<< numKeys << " ] " << *reinterpret_cast<const size_t *>(iter->key().data()) << " : ";
            // for (size_t i = 0; i < iter->key().size(); ++i) {
            //   std::cout << std::hex << std::setw(2) << (int)iter->key().data()[i] << " ";
            // }
            // std::cout << std::endl;
            ++numKeys;
          }
          return numKeys;
        }

        TensorShape value_shape() const override { return valueShape; }

        /* --- LOOKUP --------------------------------------------------------------------------- */
      protected:
        rocksdb::ColumnFamilyHandle *GetColumnHandle() const {
          rocksdb::ColumnFamilyHandle *const colHandle = link->getColumn(embeddingName);
          if (colHandle != prevColHandle) {
            std::fill(colHandleCache.begin(), colHandleCache.end(), colHandle);
            prevColHandle = colHandle;
          }
          return colHandle;
        }

      public:
        Status Clear(OpKernelContext *ctx) override { return link->deleteColumn(embeddingName); }

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

          rocksdb::ColumnFamilyHandle *const colHandle = GetColumnHandle();

          const size_t numKeys = keys.dim_size(0);
          const size_t numValues = values->dim_size(0);
          if (numKeys != numValues) {
            return errors::InvalidArgument(
              "First dimension of the key and value tensors does not match!"
            );
          }
          const size_t valuesPerDim0 = values->NumElements() / numValues;

          const K *k = static_cast<K *>(keys.data());
          const K *const kEnd = &k[numKeys];

          V *const v = static_cast<V *>(values->data());
          size_t vOffset = 0;

          const V *const d = static_cast<V *>(default_value.data());
          const size_t dSize = default_value.NumElements();

          if (dSize % valuesPerDim0 != 0) {
            return errors::InvalidArgument(
              "The shapes of the values and default_value tensors are not compatible."
            );
          }

          if (numKeys < RDB_BATCH_MODE_MIN_QUERY_SIZE) {
            rocksdb::Slice kSlice;

            for (; k != kEnd; ++k, vOffset += valuesPerDim0) {
              assignSlice(kSlice, k);
              std::string vSlice;
              auto status = colHandle
                ? (*link)->Get(readOptions, colHandle, kSlice, &vSlice)
                : rocksdb::Status::NotFound();
              if (status.ok()) {
                copyToTensor(&v[vOffset], vSlice, valuesPerDim0);
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
            // There is no point in filling this vector time and again as long as it is big enough.
            if (colHandleCache.size() < numKeys) {
              colHandleCache.insert(
                colHandleCache.end(), numKeys - colHandleCache.size(), prevColHandle
              );
            }

            // Query all keys using a single Multi-Get.
            std::vector<std::string> vSlices;
            std::vector<rocksdb::Slice> kSlices(numKeys);
            for (size_t i = 0; i < numKeys; ++i) {
              assignSlice(kSlices[i], &k[i]);
            }
            const std::vector<rocksdb::Status> &statuses = colHandle
              ? (*link)->MultiGet(readOptions, colHandleCache, kSlices, &vSlices)
              : std::vector<rocksdb::Status>(numKeys, rocksdb::Status::NotFound());
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
                copyToTensor(&v[vOffset], vSlice, valuesPerDim0);
              }
              else if (status.IsNotFound()) {
                std::copy_n(&d[vOffset % dSize], valuesPerDim0, &v[vOffset]);
              }
              else {
                throw std::runtime_error(status.getState());
              }
            }
          }

          // TODO: Instead of hard failing, return proper error code?!
          return Status::OK();
        }

        Status Insert(OpKernelContext *ctx, const Tensor &keys, const Tensor &values) override {
          if (keys.dtype() != key_dtype() || values.dtype() != value_dtype()) {
            return errors::InvalidArgument("Tensor dtypes are incompatible!");
          }

          rocksdb::ColumnFamilyHandle *const colHandle = GetColumnHandle();
          if (readOnly || !colHandle) {
            return errors::PermissionDenied("Cannot insert in read_only mode.");
          }

          const int64 numKeys = keys.dim_size(0);
          const int64 numValues = values.dim_size(0);
          if (numKeys != numValues) {
            return errors::InvalidArgument(
              "First dimension of the key and value tensors does not match!"
            );
          }
          const int64 valuesPerDim0 = values.NumElements() / numValues;

          const K *k = static_cast<K *>(keys.data());
          const K *const kEnd = &k[numKeys];

          const V *v = static_cast<V *>(values.data());

          rocksdb::Slice kSlice;
          rocksdb::PinnableSlice vSlice;

          if (numKeys < RDB_BATCH_MODE_MIN_QUERY_SIZE) {
            for (; k != kEnd; ++k, v += valuesPerDim0) {
              assignSlice(kSlice, k);
              assignSlice(vSlice, v, valuesPerDim0);
              RDB_OK((*link)->Put(writeOptions, colHandle, kSlice, vSlice));
            }
          }
          else {
            rocksdb::WriteBatch batch;
            for (; k != kEnd; ++k, v += valuesPerDim0) {
              assignSlice(kSlice, k);
              assignSlice(vSlice, v, valuesPerDim0);
              RDB_OK(batch.Put(colHandle, kSlice, vSlice));
            }
            RDB_OK((*link)->Write(writeOptions, &batch));
          }

          // Handle interval flushing.
          writeOpCount += 1;
          if (writeOpCount % flushInterval == 0) {
            (*link)->FlushWAL(true);
          }

          return Status::OK();
        }

        Status Remove(OpKernelContext *ctx, const Tensor &keys) override {
          if (keys.dtype() != key_dtype()) {
            return errors::InvalidArgument("Tensor dtypes are incompatible!");
          }

          rocksdb::ColumnFamilyHandle *const colHandle = GetColumnHandle();
          if (readOnly || !colHandle) {
            return errors::PermissionDenied("Cannot remove in read_only mode.");
          }

          const int64 numKeys = keys.dim_size(0);
          const K *k = static_cast<K *>(keys.data());
          const K *const kEnd = &k[numKeys];

          rocksdb::Slice kSlice;

          if (numKeys < RDB_BATCH_MODE_MIN_QUERY_SIZE) {
            for (; k != kEnd; ++k) {
              assignSlice(kSlice, k);
              RDB_OK((*link)->Delete(writeOptions, colHandle, kSlice));
            }
          }
          else {
            rocksdb::WriteBatch batch;
            for (; k != kEnd; ++k) {
              assignSlice(kSlice, k);
              RDB_OK(batch.Delete(colHandle, kSlice));
            }
            RDB_OK((*link)->Write(writeOptions, &batch));
          }

          // Handle interval flushing.
          writeOpCount += 1;
          if (writeOpCount % flushInterval == 0) {
            (*link)->FlushWAL(true);
          }

          return Status::OK();
        }

        /* --- IMPORT / EXPORT ------------------------------------------------------------------ */
        Status ExportValues(OpKernelContext *ctx) override {
          // Create file header.
          std::ofstream file(RDB_EXPORT_PATH, std::ofstream::binary);
          if (!file) {
            return errors::Unknown("Could not open dump file.");
          }
          file.write(
            reinterpret_cast<const char *>(&RDB_EXPORT_FILE_MAGIC),
            sizeof(RDB_EXPORT_FILE_MAGIC)
          );
          file.write(
            reinterpret_cast<const char *>(&RDB_EXPORT_FILE_VERSION),
            sizeof(RDB_EXPORT_FILE_VERSION)
          );

          // Iterate through entries one-by-one and append them to the file.
          rocksdb::ColumnFamilyHandle *const colHandle = GetColumnHandle();
          std::unique_ptr<rocksdb::Iterator> iter((*link)->NewIterator(readOptions, colHandle));
          iter->SeekToFirst();

          for (; iter->Valid(); iter->Next()) {
            const auto &kSlice = iter->key();
            if (kSlice.size() > std::numeric_limits<uint8_t>::max()) {
              throw std::runtime_error(
                "A key in the database is too long. Has the database been tampered with?"
              );
            }
            const auto kSize = static_cast<uint8_t>(kSlice.size());
            file.write(reinterpret_cast<const char *>(&kSize), sizeof(kSize));
            file.write(kSlice.data(), kSize);

            const auto vSlice = iter->value();
            if (vSlice.size() > std::numeric_limits<uint32_t>::max()) {
              throw std::runtime_error(
                "A value in the database is too large. Has the database been tampered with?"
              );
            }
            const auto vSize = static_cast<uint32_t>(vSlice.size());
            file.write(reinterpret_cast<const char *>(&vSize), sizeof(vSize));
            file.write(vSlice.data(), vSize);
          }

          return Status::OK();
        }

        Status ImportValues(
          OpKernelContext *ctx, const Tensor &keys, const Tensor &values
        ) override {
          static const Status errorEOF(error::Code::OUT_OF_RANGE, "Unexpected end of file.");

          // Make sure the column family is clean.
          const auto &clearStatus = Clear(ctx);
          if (!clearStatus.ok()) {
            return clearStatus;
          }

          // Parse header.
          std::ifstream file(RDB_EXPORT_PATH, std::ifstream::binary);
          if (!file) {
            return Status(error::Code::NOT_FOUND, "Could not open dump file.");
          }
          uint32_t magic;
          if (!file.read(reinterpret_cast<char *>(&magic), sizeof(magic))) {
            return errorEOF;
          }
          uint32_t version;
          if (!file.read(reinterpret_cast<char *>(&version), sizeof(version))) {
            return errorEOF;
          }
          if (magic != RDB_EXPORT_FILE_MAGIC || version != RDB_EXPORT_FILE_VERSION) {
            return Status(error::Code::INTERNAL, "Unsupported file-type.");
          }

          // Read payload ans subsequently populate column family.
          rocksdb::ColumnFamilyHandle *const colHandle = GetColumnHandle();
          if (readOnly || !colHandle) {
            return Status(error::Code::PERMISSION_DENIED, "Cannot import in read_only mode.");
          }

          rocksdb::WriteBatch batch;

          std::string k;
          std::string v;

          while (!file.eof()) {
            // Read key.
            uint8_t kSize;
            if (!file.read(reinterpret_cast<char *>(&kSize), sizeof(kSize))) {
              return errorEOF;
            }
            k.resize(kSize);
            if (!file.read(&k.front(), kSize)) {
              return errorEOF;
            }

            // Read value.
            uint32_t vSize;
            if (!file.read(reinterpret_cast<char *>(&vSize), sizeof(vSize))) {
              return errorEOF;
            }
            v.resize(vSize);
            if (!file.read(&v.front(), vSize)) {
              return errorEOF;
            }

            // Append to batch.
            RDB_OK(batch.Put(colHandle, k, v));

            // If batch reached target size, write to database.
            if ((batch.Count() % RDB_BATCH_MODE_MAX_QUERY_SIZE) == 0) {
              RDB_OK((*link)->Write(writeOptions, &batch));
              batch.Clear();
            }
          }

          // Write remaining entries, if any.
          if (batch.Count()) {
            RDB_OK((*link)->Write(writeOptions, &batch));
          }

          // Reset interval flushing.
          writeOpCount = 0;
          (*link)->FlushWAL(true);

          return Status::OK();
        }

      protected:
        TensorShape valueShape;
        std::string path;
        std::string embeddingName;
        bool readOnly;
        bool estimateSize;
        size_t flushInterval;
        size_t writeOpCount;
        RocksDBLink *link;
        rocksdb::ReadOptions readOptions;
        rocksdb::WriteOptions writeOptions;
        rocksdb::FlushOptions flushOptions;

        mutable rocksdb::ColumnFamilyHandle *prevColHandle;
        mutable std::vector<rocksdb::ColumnFamilyHandle *> colHandleCache;
      };

      #undef RDB_OK

      /* --- KERNEL REGISTRATION ---------------------------------------------------------------- */
      #define RDB_REGISTER_KERNEL_BUILDER(key_dtype, value_dtype)                                \
        REGISTER_KERNEL_BUILDER(                                                                 \
          Name(PREFIX_OP_NAME(RocksdbTableOfTensors))                                            \
            .Device(DEVICE_CPU)                                                                  \
            .TypeConstraint<key_dtype>("key_dtype")                                              \
            .TypeConstraint<value_dtype>("value_dtype"),                                         \
          RocksDBTableOp<RocksDBTableOfTensors<key_dtype, value_dtype>, key_dtype, value_dtype>  \
        )

      RDB_REGISTER_KERNEL_BUILDER(int32, bool);
      RDB_REGISTER_KERNEL_BUILDER(int32, int8);
      RDB_REGISTER_KERNEL_BUILDER(int32, int16);
      RDB_REGISTER_KERNEL_BUILDER(int32, int32);
      RDB_REGISTER_KERNEL_BUILDER(int32, int64);
      RDB_REGISTER_KERNEL_BUILDER(int64, Eigen::half);
      RDB_REGISTER_KERNEL_BUILDER(int32, float);
      RDB_REGISTER_KERNEL_BUILDER(int32, double);
      RDB_REGISTER_KERNEL_BUILDER(int32, tstring);

      RDB_REGISTER_KERNEL_BUILDER(int64, bool);
      RDB_REGISTER_KERNEL_BUILDER(int64, int8);
      RDB_REGISTER_KERNEL_BUILDER(int64, int16);
      RDB_REGISTER_KERNEL_BUILDER(int64, int32);
      RDB_REGISTER_KERNEL_BUILDER(int64, int64);
      RDB_REGISTER_KERNEL_BUILDER(int64, Eigen::half);
      RDB_REGISTER_KERNEL_BUILDER(int64, float);
      RDB_REGISTER_KERNEL_BUILDER(int64, double);
      RDB_REGISTER_KERNEL_BUILDER(int64, tstring);

      RDB_REGISTER_KERNEL_BUILDER(tstring, bool);
      RDB_REGISTER_KERNEL_BUILDER(tstring, int8);
      RDB_REGISTER_KERNEL_BUILDER(tstring, int16);
      RDB_REGISTER_KERNEL_BUILDER(tstring, int32);
      RDB_REGISTER_KERNEL_BUILDER(tstring, int64);
      RDB_REGISTER_KERNEL_BUILDER(tstring, Eigen::half);
      RDB_REGISTER_KERNEL_BUILDER(tstring, float);
      RDB_REGISTER_KERNEL_BUILDER(tstring, double);
      RDB_REGISTER_KERNEL_BUILDER(tstring, tstring);

      #undef RDB_TABLE_REGISTER_KERNEL_BUILDER
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

        auto *rocksTable = dynamic_cast<ClearableLookupInterface *>(table);

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
