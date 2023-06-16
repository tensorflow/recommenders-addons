/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <stddef.h>
#include <stdio.h>
#include <string>
#include "merlin/types.cuh"

namespace nv {
namespace merlin {

/**
 * The KV file on local file system. It only save/load keys and vectors
 * between table and file. `metas` are ignored in it since absolute
 * values of metas are commonly time-variant, while the time interval
 * between save/load calling is not deterministic, in default case. If
 * other specified rules are required, the BaseKVFile could be inherited
 * to implement customized read/write rules. The LocalKVFile uses compact,
 * consecutive binary format, where keys, values, and metas are stored in
 * seperated paths.
 *
 * @tparam K The data type of the key.
 * @tparam V The data type of the vector's elements.
 *         The item data type should be a basic data type of C++/CUDA.
 * @tparam M The data type for `meta`.
 *           The currently supported data type is only `uint64_t`.
 *
 */
template <class K, class V, class M>
class LocalKVFile : public BaseKVFile<K, V, M> {
 public:
  LocalKVFile() : keys_fp_(nullptr), values_fp_(nullptr), metas_fp_(nullptr) {}

  ~LocalKVFile() { close(); }

  /**
   * @brief Open the file from local path. A LocalKVFile can only be
   * read or written when it stays opened.
   *
   * @param keys_path Path to file to store keys.
   * @param values_path Path to file to store values.
   * @param metas_path Path to file to store metas.
   * @params mode The mode to the file. The mode follows glibc style
   *              and behavior like fopen.
   */
  bool open(const std::string& keys_path, const std::string& values_path,
            const std::string& metas_path, const char* mode) {
    close();
    keys_fp_ = fopen(keys_path.c_str(), mode);
    if (!keys_fp_) {
      return false;
    }
    values_fp_ = fopen(values_path.c_str(), mode);
    if (!values_fp_) {
      close();
      return false;
    }
    metas_fp_ = fopen(metas_path.c_str(), mode);
    if (!metas_fp_) {
      close();
      return false;
    }
    return true;
  }

  /**
   * @brief Close the file from open status and release fd(s) on files
   * of keys, values, and metas.
   */
  void close() noexcept {
    if (keys_fp_) {
      fclose(keys_fp_);
      keys_fp_ = nullptr;
    }
    if (values_fp_) {
      fclose(values_fp_);
      values_fp_ = nullptr;
    }
    if (metas_fp_) {
      fclose(metas_fp_);
      metas_fp_ = nullptr;
    }
  }

  /**
   * Read from file and fill into the keys, values, and metas buffer.
   * When calling save/load method from table, it can assume that the
   * received buffer of keys, vectors, and metas are automatically
   * pre-allocated.
   *
   * @param n The number of KV pairs expect to read. `int64_t` was used
   *          here to adapt to various filesytem and formats.
   * @param dim The dimension of the `vectors`.
   * @param keys The pointer to received buffer for keys.
   * @param vectors The pointer to received buffer for vectors.
   * @param metas The pointer to received buffer for metas.
   *
   * @return Number of KV pairs have been successfully read.
   */
  size_t read(const size_t n, const size_t dim, K* keys, V* vectors,
              M* metas) override {
    size_t nread_keys =
        fread(keys, sizeof(K), static_cast<size_t>(n), keys_fp_);
    size_t nread_vecs =
        fread(vectors, sizeof(V) * dim, static_cast<size_t>(n), values_fp_);
    size_t nread_metas =
        fread(metas, sizeof(M), static_cast<size_t>(n), metas_fp_);
    if (nread_keys != nread_vecs || nread_keys != nread_metas) {
      return 0;
    }
    return nread_keys;
  }

  /**
   * Write keys, values, metas from table to the file.
   *
   * @param n The number of KV pairs to be written. `int64_t` was used
   *          here to adapt to various filesytem and formats.
   * @param dim The dimension of the `vectors`.
   * @param keys The keys will be written to file.
   * @param vectors The vectors of values will be written to file.
   * @param metas The metas will be written to file.
   *
   * @return Number of KV pairs have been successfully written.
   */
  size_t write(const size_t n, const size_t dim, const K* keys,
               const V* vectors, const M* metas) override {
    size_t nwritten_keys =
        fwrite(keys, sizeof(K), static_cast<size_t>(n), keys_fp_);
    size_t nwritten_vecs =
        fwrite(vectors, sizeof(V) * dim, static_cast<size_t>(n), values_fp_);
    size_t nwritten_metas =
        fwrite(metas, sizeof(M), static_cast<size_t>(n), metas_fp_);
    if (nwritten_keys != nwritten_vecs || nwritten_keys != nwritten_metas) {
      return 0;
    }
    return nwritten_keys;
  }

 private:
  FILE* keys_fp_;
  FILE* values_fp_;
  FILE* metas_fp_;
};

}  // namespace merlin
}  // namespace nv
