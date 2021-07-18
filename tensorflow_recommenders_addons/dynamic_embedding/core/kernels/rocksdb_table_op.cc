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
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/utils.h"
#include "rocksdb_table_op.h"
#include "rocksdb/db.h"

namespace tensorflow {
  namespace recommenders_addons {
    namespace rocksdb_lookup {

      static const int64 BATCH_MODE_MIN_QUERY_SIZE = 2;
      static const uint32_t BATCH_MODE_MAX_QUERY_SIZE = 128;
      static const uint32_t EXPORT_FILE_MAGIC= ( // TODO: Little endian / big endian conversion?
        (static_cast<uint32_t>('T') << 0) |
        (static_cast<uint32_t>('F') << 8) |
        (static_cast<uint32_t>('K') << 16) |
        (static_cast<uint32_t>('V') << 24)
      );
      static const uint32_t EXPORT_FILE_VERSION = 1;

      // Note: Works for rocksdb::Status and tensorflow::Status.
      #define RDB_OK(EXPR)                            \
        do {                                          \
          const auto& s = EXPR;                       \
          if (!s.ok()) {                              \
            throw std::runtime_error(s.ToString());   \
          }                                           \
        } while (0)

      template<class T>
      void copyToTensor(T *dst, const std::string &slice, const int64 &numValues) {
        if (slice.size() != numValues * sizeof(T)) {
          std::stringstream msg;
          msg << "Expected " << numValues * sizeof(T)
              << " bytes, but " << slice.size()
              << " bytes were returned by RocksDB.";
          throw std::runtime_error(msg.str());
        }
        memcpy(dst, slice.data(), slice.size());
      }

      template<>
      void copyToTensor<tstring>(tstring *dst, const std::string &slice, const int64 &numValues) {
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
      void assignSlice(rocksdb::Slice &dst, const T &src) {
        dst.data_ = reinterpret_cast<const char *>(&src);
        dst.size_ = sizeof(T);
      }

      template<>
      void assignSlice<tstring>(rocksdb::Slice &dst, const tstring &src) {
        dst.data_ = src.data();
        dst.size_ = src.size();
      }

      template<class T>
      void assignSlice(rocksdb::PinnableSlice &dst, const T *src, const int64 numValues) {
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

      template<class K, class V>
      class RocksDBTableOfTensors : public ClearableLookupInterface {
      public:
        /* --- BASE INTERFACE ------------------------------------------------------------------- */
        RocksDBTableOfTensors(OpKernelContext *ctx, OpKernel *kernel) {
          OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "value_shape", &valueShape));
          OP_REQUIRES(ctx, TensorShapeUtils::IsVector(valueShape), errors::InvalidArgument(
            "Default value must be a vector, got shape ", valueShape.DebugString()
          ));

          std::string dbPath;
          OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "database_path", &dbPath));

          OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "embedding_name", &embeddingName));

          OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "read_only", &readOnly));

          rocksdb::Options options;
          options.create_if_missing = !readOnly;

          // Create or connect to the RocksDB database.
          std::vector<std::string> colFamilies;
          RDB_OK(rocksdb::DB::ListColumnFamilies(options, dbPath, &colFamilies));

          colIndex = 0;
          bool colFamilyExists = false;
          std::vector<rocksdb::ColumnFamilyDescriptor> colDescriptors;
          for (const auto& cf : colFamilies) {
            colDescriptors.emplace_back(cf, rocksdb::ColumnFamilyOptions());
            colFamilyExists |= cf == embeddingName;
            if (!colFamilyExists) {
              ++colIndex;
            }
          }

          db = nullptr;
          if (readOnly) {
            RDB_OK(rocksdb::DB::OpenForReadOnly(options, dbPath, colDescriptors, &colHandles, &db));
          }
          else {
            RDB_OK(rocksdb::DB::Open(options, dbPath, colDescriptors, &colHandles, &db));
          }

          // If desired column family does not exist yet, create it.
          if (!colFamilyExists) {

          }
        }

        ~RocksDBTableOfTensors() override {
          for (auto ch : colHandles) {
            db->DestroyColumnFamilyHandle(ch);
          }
          colHandles.clear();
          if (db) {
            delete db;
            db = nullptr;
          }
        }

        DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }

        DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }

        TensorShape key_shape() const override { return TensorShape(); }

        size_t size() const override { return 0; }

        TensorShape value_shape() const override { return valueShape; }

        /* --- LOOKUP --------------------------------------------------------------------------- */
        rocksdb::ColumnFamilyHandle* GetOrCreateColumnHandle() {
          if (colIndex >= colHandles.size()) {
            if (readOnly) {
              return nullptr;
            }
            rocksdb::ColumnFamilyHandle *colHandle;
            RDB_OK(db->CreateColumnFamily(
              rocksdb::ColumnFamilyOptions(), embeddingName, &colHandle
            ));
            colHandles.push_back(colHandle);
          }
          return colHandles[colIndex];
        }

        Status Clear(OpKernelContext *ctx) override {
          colHandleCache.clear();

          // Correct behavior if clear invoked multiple times.
          if (colIndex < colHandles.size()) {
            if (readOnly) {
              return Status(error::Code::PERMISSION_DENIED, "Cannot clear in read_only mode.");
            }
            RDB_OK(db->DropColumnFamily(colHandles[colIndex]));
            RDB_OK(db->DestroyColumnFamilyHandle(colHandles[colIndex]));
            colHandles.erase(colHandles.begin() + colIndex);
          }

          // Create substitute in-place.
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
            return Status(error::Code::INVALID_ARGUMENT, "Tensor dtypes are incompatible!");
          }

          rocksdb::ColumnFamilyHandle *const colHandle = GetOrCreateColumnHandle();

          const size_t numKeys = keys.dim_size(0);
          const size_t numValues = values->dim_size(0);
          if (numKeys != numValues) {
            return Status(
              error::Code::INVALID_ARGUMENT,
              "First dimension of the key and value tensors does not match!"
            );
          }
          const int64 valuesPerDim0 = values->NumElements() / numValues;

          const K *k = static_cast<K *>(keys.data());
          const K *const kEnd = &k[numKeys];

          V *const v = static_cast<V *>(values->data());
          int64 vOffset = 0;

          const V *const d = static_cast<V *>(default_value.data());
          const int64 dSize = default_value.NumElements();

          if (dSize % valuesPerDim0 != 0) {
            throw std::runtime_error(
              "The shapes of the values and default_value tensors are not compatible."
            );
          }

          if (numKeys < BATCH_MODE_MIN_QUERY_SIZE) {
            rocksdb::Slice kSlice;

            for (; k != kEnd; ++k, vOffset += valuesPerDim0) {
              assignSlice(kSlice, k);
              std::string vSlice;
              auto status = colHandle
                ? db->Get(readOptions, colHandle, kSlice, &vSlice)
                : rocksdb::Status::NotFound();
              if (status.ok()) {
                copyToTensor(&v[vOffset], vSlice, valuesPerDim0);
              }
              else if (status.IsNotFound()) {
                std::copy_n(&d[vOffset % dSize], valuesPerDim0, &v[vOffset]);
              }
              else {
                throw std::runtime_error(status.ToString());
              }
            }
          }
          else {
            // There is no point in filling this vector time and again as long as it is big enough.
            while (colHandleCache.size() < numKeys) {
              colHandleCache.push_back(colHandle);
            }

            // Query all keys using a single Multi-Get.
            std::vector<std::string> vSlices;
            std::vector<rocksdb::Slice> kSlices(numKeys);
            for (size_t i = 0; i < numKeys; ++i) {
              assignSlice(kSlices[i], k[i]);
            }
            const std::vector<rocksdb::Status> &statuses = colHandle
              ? db->MultiGet(readOptions, colHandles, kSlices, &vSlices)
              : std::vector<rocksdb::Status>(numKeys, rocksdb::Status::NotFound());
            if (statuses.size() != numKeys) {
              std::stringstream msg;
              msg << "Requested " << numKeys << " keys, but only got " << statuses.size()
                  << " responses.";
              throw std::runtime_error(msg.str());
            }

            // Process results.
            for (size_t i = 0; i < numKeys; ++i, vOffset += valuesPerDim0) {
              const auto& status = statuses[i];
              const auto& vSlice = vSlices[i];

              if (status.ok()) {
                copyToTensor(&v[vOffset], vSlice, valuesPerDim0);
              }
              else if (status.IsNotFound()) {
                std::copy_n(&d[vOffset % dSize], valuesPerDim0, &v[vOffset]);
              }
              else {
                throw std::runtime_error(status.ToString());
              }
            }
          }

          // TODO: Instead of hard failing, return proper error code?!
          return Status::OK();
        }

        Status Insert(OpKernelContext *ctx, const Tensor &keys, const Tensor &values) override {
          if (keys.dtype() != key_dtype() || values.dtype() != value_dtype()) {
            return Status(error::Code::INVALID_ARGUMENT, "Tensor dtypes are incompatible!");
          }

          rocksdb::ColumnFamilyHandle *const colHandle = GetOrCreateColumnHandle();
          if (!colHandle || readOnly) {
            return Status(error::Code::PERMISSION_DENIED, "Cannot insert in read_only mode.");
          }

          const int64 numKeys = keys.dim_size(0);
          const int64 numValues = values.dim_size(0);
          if (numKeys != numValues) {
            return Status(
              error::Code::INVALID_ARGUMENT,
              "First dimension of the key and value tensors does not match!"
            );
          }
          const int64 valuesPerDim0 = values.NumElements() / numValues;

          const K *k = static_cast<K *>(keys.data());
          const K *const kEnd = &k[numKeys];

          const V *v = static_cast<V *>(values.data());

          rocksdb::Slice kSlice;
          rocksdb::PinnableSlice vSlice;

          if (numKeys < BATCH_MODE_MIN_QUERY_SIZE) {
            for (; k != kEnd; ++k, v += valuesPerDim0) {
              assignSlice(kSlice, k);
              assignSlice(vSlice, v, valuesPerDim0);
              RDB_OK(db->Put(writeOptions, colHandle, kSlice, vSlice));
            }
          }
          else {
            rocksdb::WriteBatch batch;
            for (; k != kEnd; ++k, v += valuesPerDim0) {
              assignSlice(kSlice, k);
              assignSlice(vSlice, v, valuesPerDim0);
              RDB_OK(batch.Put(colHandle, kSlice, vSlice));
            }
            RDB_OK(db->Write(writeOptions, &batch));
          }

          // TODO: Instead of hard failing, return proper error code?!
          return Status::OK();
        }

        Status Remove(OpKernelContext *ctx, const Tensor &keys) override {
          if (keys.dtype() != key_dtype()) {
            return Status(error::Code::INVALID_ARGUMENT, "Tensor dtypes are incompatible!");
          }

          rocksdb::ColumnFamilyHandle *const colHandle = GetOrCreateColumnHandle();
          if (!colHandle || readOnly) {
            return Status(error::Code::PERMISSION_DENIED, "Cannot remove in read_only mode.");
          }

          const int64 numKeys = keys.dim_size(0);
          const K *k = static_cast<K *>(keys.data());
          const K *const kEnd = &k[numKeys];

          rocksdb::Slice kSlice;

          if (numKeys < BATCH_MODE_MIN_QUERY_SIZE) {
            for (; k != kEnd; ++k) {
              assignSlice(kSlice, k);
              RDB_OK(db->Delete(writeOptions, colHandle, kSlice));
            }
          }
          else {
            rocksdb::WriteBatch batch;
            for (; k != kEnd; ++k) {
              assignSlice(kSlice, k);
              RDB_OK(batch.Delete(colHandle, kSlice));
            }
            RDB_OK(db->Write(writeOptions, &batch));
          }

          // TODO: Instead of hard failing, return proper error code?!
          return Status::OK();
        }

        /* --- IMPORT / EXPORT ------------------------------------------------------------------ */
        Status ExportValues(OpKernelContext *ctx) override {
          // Create file header.
          std::ofstream file("/tmp/db.dump", std::ofstream::binary);
          if (!file) {
            return Status(error::Code::UNKNOWN, "Could not open dump file.");
          }
          file.write(
            reinterpret_cast<const char *>(&EXPORT_FILE_MAGIC), sizeof(EXPORT_FILE_MAGIC)
          );
          file.write(
            reinterpret_cast<const char *>(&EXPORT_FILE_VERSION), sizeof(EXPORT_FILE_VERSION)
          );

          // Iterate through entries one-by-one and append them to the file.
          rocksdb::ColumnFamilyHandle *const colHandle = GetOrCreateColumnHandle();
          std::unique_ptr<rocksdb::Iterator> iter(db->NewIterator(readOptions, colHandle));
          iter->SeekToFirst();

          for (; iter->Valid(); iter->Next()) {
            const auto& kSlice = iter->key();
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
          RDB_OK(Clear(ctx));

          // Parse header.
          std::ifstream file("/tmp/db.dump", std::ifstream::binary);
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
          if (magic != EXPORT_FILE_MAGIC || version != EXPORT_FILE_VERSION) {
            return Status(error::Code::INTERNAL, "Unsupported file-type.");
          }

          // Read payload ans subsequently populate column family.
          rocksdb::ColumnFamilyHandle *const colHandle = GetOrCreateColumnHandle();
          if (!colHandle || readOnly) {
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
            if ((batch.Count() % BATCH_MODE_MAX_QUERY_SIZE) == 0) {
              RDB_OK(db->Write(writeOptions, &batch));
              batch.Clear();
            }
          }

          // Write remaining entries, if any.
          if (batch.Count()) {
            RDB_OK(db->Write(writeOptions, &batch));
          }

          return Status::OK();
        }

      protected:
        TensorShape valueShape;
        std::string embeddingName;
        bool readOnly;
        rocksdb::DB *db;
        std::vector<rocksdb::ColumnFamilyHandle*> colHandles;
        size_t colIndex;
        rocksdb::ReadOptions readOptions;
        rocksdb::WriteOptions writeOptions;

        std::vector<rocksdb::ColumnFamilyHandle*> colHandleCache;
      };

      #undef RDB_OK

      /* --- KERNEL REGISTRATION ---------------------------------------------------------------- */
      #define RDB_REGISTER_KERNEL_BUILDER(key_dtype, value_dtype)                                \
        REGISTER_KERNEL_BUILDER(                                                                 \
          Name(PREFIX_OP_NAME(RocksDBTableOfTensors))                                            \
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

      REGISTER_KERNEL_BUILDER(
        Name(PREFIX_OP_NAME(RocksDBTableClear)).Device(DEVICE_CPU), RocksDBTableClear
      );
      REGISTER_KERNEL_BUILDER(
        Name(PREFIX_OP_NAME(RocksDBTableExport)).Device(DEVICE_CPU), RocksDBTableExport
      );
      REGISTER_KERNEL_BUILDER(
        Name(PREFIX_OP_NAME(RocksDBTableFind)).Device(DEVICE_CPU), RocksDBTableFind
      );
      REGISTER_KERNEL_BUILDER(
        Name(PREFIX_OP_NAME(RocksDBTableImport)).Device(DEVICE_CPU), RocksDBTableImport
      );
      REGISTER_KERNEL_BUILDER(
        Name(PREFIX_OP_NAME(RocksDBTableInsert)).Device(DEVICE_CPU), RocksDBTableInsert
      );
      REGISTER_KERNEL_BUILDER(
        Name(PREFIX_OP_NAME(RocksDBTableRemove)).Device(DEVICE_CPU), RocksDBTableRemove
      );
      REGISTER_KERNEL_BUILDER(
        Name(PREFIX_OP_NAME(RocksDBTableSize)).Device(DEVICE_CPU), RocksDBTableSize
      );

    }  // namespace rocksdb_lookup
  }  // namespace recommenders_addons
}  // namespace tensorflow
