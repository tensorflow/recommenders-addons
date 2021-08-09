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
#pragma once
#include <aio.h>
#include <stdlib.h>
#include <sw/redis++/connection.h>
#include <sw/redis++/connection_pool.h>
#include <sw/redis++/redis++.h>
#include <sw/redis++/sentinel.h>
#include <sys/stat.h>
#include <unistd.h>
#include <x86intrin.h>

#include <cmath>
#include <iostream>

#include "md5.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace recommenders_addons {
namespace redis_connection {

const static unsigned hardware_concurrency_ =
    std::thread::hardware_concurrency();

/*
if v=0 or 1, return itself. if v<0, v=abs(v).
*/
inline unsigned round_next_power_two_bitlen(int v) {
  v = abs(v);
  if (v == 1 || v == 0) {
    return 0;
  }

  if ((v | 1) == 1) return v;
  // prevent v is already a power_two
  --v;
  // How many bits for storing v at least.
  int bitlen = 1 + static_cast<int>(std::log2(v));
  // v = 1 << bitlen; //number_round_next_power_two
  return bitlen;
}

inline unsigned long get_file_size(const std::string path) {
  unsigned long filesize = -1;
  struct stat statbuff;
  if (stat(path.data(), &statbuff) < 0) {
    throw(std::invalid_argument("The file " + path + " does not exist"));
    return filesize;
  } else {
    filesize = statbuff.st_size;
  }
  return filesize;
}

inline int createDirectory(const std::string path) {
  int len = path.length();
  char tmpDirPath[1024] = {0};
  for (int i = 0; i < len; i++) {
    tmpDirPath[i] = path[i];
    if (tmpDirPath[i] == '\\' || tmpDirPath[i] == '/') {
      if (access(tmpDirPath, 0) == -1) {
        int ret = mkdir(tmpDirPath, S_IRWXU | S_IRWXG | S_IRWXO);
        if (ret == -1) return ret;
      }
    }
  }
  return 0;
}

inline std::string check_dir(const std::string path_in) {
  std::string path(path_in);
  if (path.back() != '/') {
    path.push_back('/');
  }
  if (access(path.c_str(), 0) == -1)  // if folder doesn't exist
  {
    LOG(INFO) << "folder " << path << " doesn't exist";
    if (createDirectory(path) == 0) {
      LOG(INFO) << "folder " << path << " was created";
    } else {
      LOG(INFO) << "folder " << path << " failed to create";
    }
  }
  return path;
}

inline int ReadInt32FromEnvVar(const std::string &env_var_name,
                               const int &default_val, int *value) {
  const char *_env_var_val = getenv(env_var_name.c_str());
  if (_env_var_val == nullptr) {
    return 0;
  }
  *value = std::stoi(_env_var_val);
  if (static_cast<long long>(*value) == std::stol(_env_var_val)) {
    return 0;
  }
  *value = default_val;
  LOG(ERROR) << "Failed to parse the env-var ${" << env_var_name
             << "} into int32: " << _env_var_val
             << ". Use the default value: " << default_val;
  return -1;
}

std::array<unsigned char, 16> MD5(const std::string &src) {
  MD5_CTX ctx;
  std::array<unsigned char, 16> md;
  MD5Init(&ctx);
  MD5Update(&ctx, (unsigned char *)src.c_str(), src.size());
  MD5Final(md.data(), &ctx);
  return md;
}

enum Connection_Mode { ClusterMode = 0, SentinelMode = 1, StreamMode = 2 };

struct Redis_Connection_Params {
  int redis_connection_mode =
      1;  // ClusterMode = 0, SentinelMode = 1, StreamMode = 2
  std::string redis_master_name = "master";
  // connection_options
  std::vector<std::string> redis_host_ip = {"127.0.0.1"};
  std::vector<int> redis_host_port = {6379};
  std::string redis_password = "";
  int redis_db = 0;
  //
  int redis_connect_timeout = 1000;  // milliseconds
  int redis_socket_timeout = 1000;   // milliseconds
  // connection_pool_options
  int redis_conn_pool_size = 20;
  int redis_wait_timeout = 100000000;   // milliseconds
  int redis_connection_lifetime = 100;  // minutes
  // sentinel_connection_options
  int redis_sentinel_connect_timeout = 1000;  // milliseconds
  int redis_sentinel_socket_timeout = 1000;   // milliseconds
  //
  // Below there is user-defined parameters in this custom op, not Redis
  // setting parameters
  unsigned storage_slice =
      1;  // For deciding hash tag, which usually is how many Redis instance
          // may be used in the trainning.
  unsigned expire_model_tag_in_seconds =
      604800;  // To eliminate unwanted model versions in Redis to ensure
               // sufficient storage space.
  unsigned long long keys_sending_size =
      1024;  // Determines how many keys to send at a time
             // for performance tuning
  unsigned storage_slice_log2 = 0;  // For fast calculation.
  std::string model_tag_old =
      "test";  // old model_tag for version and any other information
  std::string model_tag_new =
      "test";  // new model_tag for version and any other information
  bool using_md5_prefix_name = false;
  std::string model_lib_abs_dir = "/tmp/";
  bool using_model_lib = true;

  Redis_Connection_Params &operator=(const Redis_Connection_Params &x) {
    redis_connection_mode = x.redis_connection_mode;
    redis_master_name = x.redis_master_name;
    redis_host_ip.assign(x.redis_host_ip.begin(), x.redis_host_ip.end());
    redis_host_port.assign(x.redis_host_port.begin(), x.redis_host_port.end());
    redis_password = x.redis_password;
    redis_db = x.redis_db;
    redis_connect_timeout = x.redis_connect_timeout;  // milliseconds
    redis_socket_timeout = x.redis_socket_timeout;    // milliseconds
    redis_conn_pool_size = x.redis_conn_pool_size;
    redis_wait_timeout = x.redis_wait_timeout;                // milliseconds
    redis_connection_lifetime = x.redis_connection_lifetime;  // minutes
    redis_sentinel_connect_timeout =
        x.redis_sentinel_connect_timeout;  // milliseconds
    redis_sentinel_socket_timeout =
        x.redis_sentinel_socket_timeout;  // milliseconds
    storage_slice_log2 =
        round_next_power_two_bitlen(x.storage_slice);  // beter for modding.
    storage_slice = 1 << storage_slice_log2;
    expire_model_tag_in_seconds = x.expire_model_tag_in_seconds > 0
                                      ? x.expire_model_tag_in_seconds
                                      : 2626560;
    model_tag_old = x.model_tag_old;
    model_tag_new = x.model_tag_new;
    using_md5_prefix_name = x.using_md5_prefix_name;
    model_lib_abs_dir = check_dir(x.model_lib_abs_dir);
    using_model_lib = x.using_model_lib;
    return *this;
  }
};

class BucketContext {
 public:
  std::unique_ptr<std::vector<const char *>> ptrs;
  std::unique_ptr<std::vector<std::size_t>> sizes;

  void HandleRelease() {
    if (this->ptrs.get()) {
      this->ptrs.reset();
    }
    if (this->sizes.get()) {
      this->sizes.reset();
    }
  }

  BucketContext() {
    this->ptrs = std::make_unique<std::vector<const char *>>(
        std::vector<const char *>());
    this->ptrs->reserve(8);
    this->sizes =
        std::make_unique<std::vector<std::size_t>>(std::vector<std::size_t>());
    this->sizes->reserve(8);
  }

  ~BucketContext() { HandleRelease(); }
};

class ThreadContext {
 public:
  std::atomic<bool> thread_occupied{false};
  std::vector<std::unique_ptr<BucketContext>> buckets;
  std::unique_ptr<std::vector<unsigned>> bucket_locs;

  void HandleReserve(const unsigned storage_slice, const unsigned vector_len,
                     const int keys_num) {
    for (size_t i = this->buckets.size(); i != storage_slice; ++i) {
      buckets.emplace_back(std::unique_ptr<BucketContext>(new BucketContext()));
    }
    for (unsigned i = 0; i < storage_slice; ++i) {
      this->buckets[i]->ptrs->clear();
      this->buckets[i]->ptrs->reserve(vector_len);
      this->buckets[i]->sizes->clear();
      this->buckets[i]->sizes->reserve(vector_len);
    }
    this->bucket_locs->reserve(keys_num);
  }

  void HandlePushBack(const unsigned bucket_num, const char *ptrs_in,
                      const std::size_t sizes_in) {
    this->buckets[bucket_num]->ptrs->emplace_back(ptrs_in);
    this->buckets[bucket_num]->sizes->emplace_back(sizes_in);
  }

  void HandleRelease() {
    if (this->bucket_locs.get()) {
      bucket_locs.reset();
    }
    for (size_t i = 0; i < this->buckets.size(); ++i) {
      if (buckets[i].get()) {
        buckets[i].reset();
      }
    }
  }

  ThreadContext() {
    this->buckets.emplace_back(
        std::unique_ptr<BucketContext>(new BucketContext()));
    this->bucket_locs =
        std::make_unique<std::vector<unsigned>>(std::vector<unsigned>());
    this->bucket_locs->reserve(8);
  }

  ~ThreadContext() { HandleRelease(); }
};

class RedisVirtualWrapper {
 protected:
  bool isRedisConnect = false;
  Redis_Connection_Params redis_connection_params;

 public:
  void set_params(struct Redis_Connection_Params &conn_params_input) {
    this->redis_connection_params = conn_params_input;
  }

  virtual void Conn() = 0;

  virtual int CheckSlicesNum(const std::string &keys_prefix_name) = 0;

  virtual size_t TableSizeInBuckets(
      const std::vector<std::string> &keys_prefix_name_slices) = 0;

  virtual void RemoveHkeysInBuckets(
      const std::vector<std::string> &keys_prefix_name_slices) = 0;

  virtual std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
  GetKeysInHkeys(const std::vector<std::string> &keys_prefix_name_slices) = 0;

  virtual void SetExpireBuckets(
      const std::vector<std::string> &keys_prefix_name_slices) = 0;

  virtual void DumpToDisk(
      const std::vector<std::string> &keys_prefix_name_slices,
      std::vector<aiocb> &wrs, const std::vector<int> &fds) = 0;

  virtual void RestoreFromDisk(
      const std::vector<std::string> &keys_prefix_name_slices,
      std::vector<aiocb> &rds, const std::vector<int> &fds,
      const std::vector<unsigned long> &buf_sizes) = 0;

  virtual void DuplicateInRedis(
      const std::vector<std::string> &keys_prefix_name_slices_old,
      const std::vector<std::string> &keys_prefix_name_slices_new) = 0;

  virtual std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
  MgetCommand(const Tensor &keys, ThreadContext *thread_context,
              const int64 begin, const int64 max_i,
              const std::vector<std::string> &keys_prefix_name_slices) = 0;

  virtual void MgetToTensor(
      Tensor *values, const Tensor &default_value, const bool is_full_default,
      ThreadContext *thread_context,
      std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
          &reply,
      const int64 begin, const int64 max_i, const int64 Velems_per_dim0) = 0;

  virtual void MsetCommand(
      const Tensor &keys, const Tensor &values, ThreadContext *thread_context,
      const int64 begin, const int64 max_i, const int64 Velems_per_dim0,
      const std::vector<std::string> &keys_prefix_name_slices) = 0;

  virtual void DelCommand(
      const Tensor &keys, ThreadContext *thread_context, const int64 begin,
      const int64 max_i,
      const std::vector<std::string> &keys_prefix_name_slices) = 0;
};

template <typename RedisInstance, typename K, typename V, typename = void>
class RedisWrapper : public RedisVirtualWrapper {};

struct VContentAndTypeSizeResult {
  size_t VTypeSize;
  const char *VContentPointer;
};

template <typename T>
constexpr inline size_t KTypeSize(const T *in) {
  return sizeof(T);
}

template <>
inline size_t KTypeSize<tstring>(const tstring *in) {
  return in->size();
}

template <typename T>
constexpr inline const char *KContentPointer(const T *in) {
  return reinterpret_cast<const char *>(in);
}

template <>
inline const char *KContentPointer<tstring>(const tstring *in) {
  return in->data();
}

template <typename T>
inline unsigned KBucketNum(const T *in, const unsigned storage_slice) {
  return static_cast<const int>(*in) & (storage_slice - 1);
}

template <>
inline unsigned KBucketNum<tstring>(const tstring *in,
                                    const unsigned storage_slice) {
  const auto tem_char = *(reinterpret_cast<const unsigned char *>(in));
  return (_mm_crc32_u8(0xffffffff, tem_char) & (storage_slice - 1));
}

template <typename T>
inline const VContentAndTypeSizeResult &VContentAndTypeSize(
    VContentAndTypeSizeResult &_VContentAndTypeSizeResult,
    const int64 Velems_per_dim0, const std::size_t &V_byte_size, const T *in,
    std::vector<char> &buff) {
  _VContentAndTypeSizeResult.VTypeSize = V_byte_size;
  _VContentAndTypeSizeResult.VContentPointer =
      reinterpret_cast<const char *>(in);
  return _VContentAndTypeSizeResult;
}

/*
Because the string tensor in tensorflow only store the pointer that points
to string value, we use a char buffer to store all string into a continuous
memory space for sending to Redis server.
[[str_size(4 bytes),string[...](ps->size() bytes)],[str_size,string[...]],[...]]
var buff is a std::vector<char> from std::vector<std::vector<char>>
*/
template <>
inline const VContentAndTypeSizeResult &VContentAndTypeSize<tstring>(
    VContentAndTypeSizeResult &_VContentAndTypeSizeResult,
    const int64 Velems_per_dim0, const std::size_t &V_byte_size,
    const tstring *in, std::vector<char> &buff) {
  const tstring *ps_end = in + Velems_per_dim0;
  unsigned tot = 0;

  const tstring *ps = in;
  for (; ps != ps_end; ++ps) {
    tot = tot + sizeof(unsigned) + ps->size();
  }

  _VContentAndTypeSizeResult.VTypeSize = tot;
  buff.reserve(tot);
  _VContentAndTypeSizeResult.VContentPointer = buff.data();

  ps = in;  // Reset the content pointer.

  unsigned tem_byte_size = 0;
  char *chars = nullptr;

  for (; ps != ps_end; ++ps) {
    tem_byte_size = static_cast<unsigned>(ps->size());
    chars = reinterpret_cast<char *>(&tem_byte_size);
    buff.insert(
        buff.end(), chars,
        chars + sizeof(unsigned));  // Firstly store how long the string is
    buff.insert(buff.end(), ps->begin(), ps->end());
  }

  return _VContentAndTypeSizeResult;
}

template <typename T>
void DefaultMemcpyToTensor(const T *const pv_raw, const T *dft,
                           const int64 Velems_per_dim0) {
  void *pv_raw_ = reinterpret_cast<void *>(const_cast<T *>(pv_raw));
  memcpy(pv_raw_, reinterpret_cast<const void *>(dft),
         Velems_per_dim0 *
             sizeof(T));  // Direct access to Tensor data in TensorFlow
}

template <>
void DefaultMemcpyToTensor<tstring>(const tstring *const pv_raw,
                                    const tstring *const dft,
                                    const int64 Velems_per_dim0) {
  const tstring *const pv_raw_end = pv_raw + Velems_per_dim0;
  tstring *pv_it = const_cast<tstring *>(pv_raw);
  const tstring *dft_it = dft;
  for (; pv_it != pv_raw_end; ++pv_it, ++dft_it) {
    *pv_it = *dft_it;
  }
}

template <typename T>
void ReplyMemcpyToKeyTensor(const T *const pk_raw, const char *str,
                            const size_t &byte_size) {
  void *pk_raw_ = reinterpret_cast<void *>(const_cast<T *>(pk_raw));
  memcpy(pk_raw_, str,
         byte_size);  // Direct access to Tensor data in TensorFlow
}

template <>
void ReplyMemcpyToKeyTensor<tstring>(const tstring *const pk_raw,
                                     const char *str, const size_t &byte_size) {
  (const_cast<tstring *const>(pk_raw))->assign(str, byte_size);
}

template <typename T>
void ReplyMemcpyToValTensor(const T *const pv_raw, const char *str,
                            const int64 Velems_per_dim0) {
  void *pv_raw_ = reinterpret_cast<void *>(const_cast<T *>(pv_raw));
  memcpy(pv_raw_, str,
         Velems_per_dim0 *
             sizeof(T));  // Direct access to Tensor data in TensorFlow
}

template <>
void ReplyMemcpyToValTensor<tstring>(const tstring *const pv_raw,
                                     const char *str,
                                     const int64 Velems_per_dim0) {
  const tstring *const pv_raw_end = pv_raw + Velems_per_dim0;
  const char *char_view = str;
  unsigned str_bytesize = 0;
  for (tstring *pv_it = const_cast<tstring *>(pv_raw); pv_it != pv_raw_end;
       ++pv_it) {
    str_bytesize = *(reinterpret_cast<const unsigned *>(char_view));
    char_view += sizeof(unsigned);
    pv_it->assign(char_view, str_bytesize);
    char_view += str_bytesize;
  }
}

}  // namespace redis_connection
}  // namespace recommenders_addons
}  // namespace tensorflow
