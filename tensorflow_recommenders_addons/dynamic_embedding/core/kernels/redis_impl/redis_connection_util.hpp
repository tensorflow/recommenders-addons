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

#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <vector>

#include "md5.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/utils.h"

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
  unsigned long filesize = 0;
  struct stat statbuff;
  if (stat(path.data(), &statbuff) < 0) {
    LOG(WARNING) << "The file " << path << " does not exist";
    return filesize;
  } else {
    filesize = statbuff.st_size;
  }
  return filesize;
}

inline int createDirectory(const std::string path) {
  size_t len = path.size();
  std::vector<char> tmpDirPath;
  tmpDirPath.resize(len);
  for (size_t i = 0; i < len; i++) {
    tmpDirPath[i] = path[i];
    if (tmpDirPath[i] == '\\' || tmpDirPath[i] == '/') {
      if (access(tmpDirPath.data(), 0) == -1) {
        int ret = mkdir(tmpDirPath.data(), S_IRWXU | S_IRWXG | S_IRWXO);
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

enum Connection_Mode { ClusterMode = 0, SentinelMode = 1, StandaloneMode = 2 };

struct Redis_Connection_Params {
  int redis_connection_mode =
      1;  // ClusterMode = 0, SentinelMode = 1, StandaloneMode = 2
  std::string redis_master_name = "master";
  // connection_options
  std::vector<std::string> redis_host_ip = {"127.0.0.1"};
  std::vector<int> redis_host_port = {6379};
  std::string redis_user = "default";
  std::string redis_password = "";
  int redis_db = 0;
  //
  bool redis_read_access_slave = false;
  bool redis_connect_keep_alive = false;
  int redis_connect_timeout = 1000;  // milliseconds
  int redis_socket_timeout = 1000;   // milliseconds
  // connection_pool_options
  int redis_conn_pool_size = 20;
  int redis_wait_timeout = 100000000;   // milliseconds
  int redis_connection_lifetime = 100;  // minutes
  // sentinel_connection_options
  std::string redis_sentinel_user = "default";
  std::string redis_sentinel_password = "";
  int redis_sentinel_connect_timeout = 1000;  // milliseconds
  int redis_sentinel_socket_timeout = 1000;   // milliseconds
  //
  // Below there is user-defined parameters in this custom op, not Redis
  // setting parameters
  int storage_slice_import = -1;
  unsigned storage_slice =
      1;  // For deciding bucket number, which usually is how
          // many Redis instance may be used in the trainning.
  bool using_hash_storage_slice =
      false;  // If True, IDs will be calculated hash(CRC32) value and then MOD
              // to decide which bucket number they belong to. If False, only
              // calculate the remainder.
  unsigned long long keys_sending_size =
      1024;  // Determines how many keys to send at a time
             // for performance tuning
  bool using_md5_prefix_name = false;
  bool redis_hash_tags_hypodispersion =
      false;  // distribution of storag_slice will be hypodispersion in 16354
              // regardless cluster slot, but still depends on
              // redis_hash_tags_import/runtime if they aren't empty.
  std::string model_tag_import =
      "test";  // old model_tag for version and any other information
  std::vector<std::string> redis_hash_tags_import =
      {};  // Deciding hash tag for every bucket, Note that the hash tag must be
           // wrapped in curly braces. For example {3560}.
  std::string model_tag_runtime =
      "test";  // new model_tag for version and any other information
  std::vector<std::string> redis_hash_tags_runtime =
      {};  // Deciding hash tag for every bucket, Note that the hash tag must be
           // wrapped in curly braces. For example {3560}.
  int expire_model_tag_in_seconds =
      604800;  // To eliminate unwanted model versions in Redis to ensure
               // sufficient storage space. It will not take effect if it is
               // less than zero.
  std::string model_lib_abs_dir = "/tmp/";
  // if table_store_mode equals 1, then it will try to save or resoter table
  // from model_lib_abs_dir which has been mounted in system
  unsigned table_store_mode = 1;
  // Saving and restoring table into ensor in TF savedmodel variable file,
  // table_store_mode = 0; Saving and restoring table into redis rdb file in
  // model_lib_abs_dir, table_store_mode = 1; Saving and restoring nothing,
  // keeping data in redis servers, table_store_mode = 2.

  Redis_Connection_Params &operator=(const Redis_Connection_Params &x) {
    redis_connection_mode = x.redis_connection_mode;
    redis_master_name = x.redis_master_name;
    redis_host_ip.assign(x.redis_host_ip.begin(), x.redis_host_ip.end());
    redis_host_port.assign(x.redis_host_port.begin(), x.redis_host_port.end());
    redis_user = x.redis_user;
    redis_password = x.redis_password;
    redis_db = x.redis_db;
    redis_read_access_slave = x.redis_read_access_slave;
    redis_connect_keep_alive = x.redis_connect_keep_alive;
    redis_connect_timeout = x.redis_connect_timeout;  // milliseconds
    redis_socket_timeout = x.redis_socket_timeout;    // milliseconds
    redis_conn_pool_size = x.redis_conn_pool_size;
    redis_wait_timeout = x.redis_wait_timeout;                // milliseconds
    redis_connection_lifetime = x.redis_connection_lifetime;  // minutes
    redis_sentinel_user = x.redis_sentinel_user;
    redis_sentinel_password = x.redis_sentinel_password;
    redis_sentinel_connect_timeout =
        x.redis_sentinel_connect_timeout;  // milliseconds
    redis_sentinel_socket_timeout =
        x.redis_sentinel_socket_timeout;  // milliseconds
    storage_slice_import =
        x.storage_slice_import >= 0 ? x.storage_slice_import : x.storage_slice;
    storage_slice = x.storage_slice;
    using_hash_storage_slice = x.using_hash_storage_slice;
    keys_sending_size = x.keys_sending_size;
    using_md5_prefix_name = x.using_md5_prefix_name;
    redis_hash_tags_hypodispersion = x.redis_hash_tags_hypodispersion;
    model_tag_import = x.model_tag_import;
    redis_hash_tags_import.assign(x.redis_hash_tags_import.begin(),
                                  x.redis_hash_tags_import.end());
    model_tag_runtime = x.model_tag_runtime;
    redis_hash_tags_runtime.assign(x.redis_hash_tags_runtime.begin(),
                                   x.redis_hash_tags_runtime.end());
    expire_model_tag_in_seconds = x.expire_model_tag_in_seconds;
    model_lib_abs_dir = check_dir(x.model_lib_abs_dir);
    table_store_mode = x.table_store_mode;
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

  void HandleClear() {
    this->ptrs->clear();
    this->sizes->clear();
  }

  void HandleReserve(const unsigned vector_len) {
    this->ptrs->reserve(vector_len);
    this->sizes->reserve(vector_len);
  }

  void HandlePushBack(const char *ptrs_in, const std::size_t sizes_in) {
    this->ptrs->emplace_back(ptrs_in);
    this->sizes->emplace_back(sizes_in);
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
      this->buckets[i]->HandleClear();
      this->buckets[i]->HandleReserve(vector_len);
    }
    this->bucket_locs->reserve(keys_num);
  }

  void HandlePushBack(const unsigned bucket_num, const char *ptrs_in,
                      const std::size_t sizes_in) {
    this->buckets[bucket_num]->HandlePushBack(ptrs_in, sizes_in);
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

typedef unsigned (*KBucketNumHandle)(uint32_t, const uint8_t *, size_t);

template <typename K, typename V>
class RedisBaseWrapper {
 protected:
  Redis_Connection_Params redis_connection_params;
  KBucketNumHandle K_bucket_num_handle;

 protected:
  template <typename RedisClient>
  inline bool RedisClusterEnabled(RedisClient redis_client) {
    auto info_cluster = redis_client->command("info", "cluster");
    if (info_cluster->len > 0) {
      auto tmp_char = strtok(info_cluster->str, "\n");
      tmp_char = strtok(NULL, "\n");
      tmp_char = strtok(tmp_char, ":");
      auto cluster_bool = strtok(NULL, ":");
      if (strcmp(cluster_bool, "1\r") == 0) {
        return true;
      } else {
        return false;
      }
    } else {
      LOG(WARNING)
          << "INFO CLUSTER has no response. Regard as a single node mode.";
      return false;
    }
  }

  template <typename ConnOpts, typename PoolOpts, typename ConnParams>
  inline void SetPublicConnParams(ConnOpts &conn_opts, PoolOpts &pool_opts,
                                  ConnParams &redis_connection_params) {
    // Redis connection options
    conn_opts.user = redis_connection_params.redis_user;
    conn_opts.password =
        redis_connection_params
            .redis_password;  // Optional. No redis_password by default.
    conn_opts.db = redis_connection_params.redis_db;
    conn_opts.keep_alive = redis_connection_params.redis_connect_keep_alive;
    conn_opts.connect_timeout = std::chrono::milliseconds(
        redis_connection_params.redis_connect_timeout);
    conn_opts.socket_timeout =
        std::chrono::milliseconds(redis_connection_params.redis_socket_timeout);
    // Redis connection pool options
    pool_opts.size = redis_connection_params.redis_conn_pool_size;
    pool_opts.wait_timeout =
        std::chrono::milliseconds(redis_connection_params.redis_wait_timeout);
    pool_opts.connection_lifetime =
        std::chrono::minutes(redis_connection_params.redis_connection_lifetime);
  }

 public:
  bool isRedisConnect = false;

 public:
  Status set_params(struct Redis_Connection_Params &conn_params_input) {
    try {
      this->redis_connection_params = conn_params_input;
    } catch (const std::exception &err) {
      return errors::Unknown(err.what());
    }
    return TFOkStatus;
  }

  Status set_K_bucket_num_handle(KBucketNumHandle function) {
    try {
      K_bucket_num_handle = function;
    } catch (const std::exception &err) {
      return errors::Unknown(err.what());
    }
    return TFOkStatus;
  }

  virtual Status Conn() = 0;

  virtual std::vector<std::string> GetKeyBucketsAndOptimizerParamsWithName(
      const std::string &keys_prefix_name, const bool only_get_buckets) = 0;

  virtual int CheckSlicesNum(const std::string &keys_prefix_name) = 0;

  virtual std::vector<std::pair<unsigned, unsigned>> ClusterNodesSlots(
      bool full_slots) = 0;

  virtual size_t TableSizeInBucket(
      const std::string &keys_prefix_name_slice) = 0;

  virtual Status RemoveHkeysInBuckets(
      const std::string &keys_prefix_name_slice) = 0;

  virtual std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>
  HscanGetKeysValsInBucket(const std::string &keys_prefix_name_slice,
                           long long *cursor, const long long count) = 0;

  virtual std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> MgetInBucket(
      const K *, const int64_t begin, const int64_t max_i,
      const std::string &keys_prefix_name_slice) = 0;

  virtual Status SetExpireBuckets(const std::string &keys_prefix_name) = 0;

  virtual Status SetPersistBuckets(const std::string &keys_prefix_name) = 0;

  virtual Status DumpToDisk(
      const std::vector<std::string> &keys_prefix_name_slices,
      std::vector<aiocb> &wrs, const std::vector<int> &fds) = 0;

  virtual Status RestoreFromDisk(
      const std::vector<std::string> &keys_prefix_name_slices,
      std::vector<aiocb> &rds, const std::vector<int> &fds,
      const std::vector<unsigned long> &buf_sizes) = 0;

  virtual Status DuplicateInRedis(
      const std::vector<std::string> &keys_prefix_name_slices_old,
      const std::vector<std::string> &keys_prefix_name_slices_new) = 0;

  virtual std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
  MgetCommand(const K *, ThreadContext *thread_context, const int64_t begin,
              const int64_t max_i,
              const std::vector<std::string> &keys_prefix_name_slices) = 0;

  virtual Status MgetToTensor(
      V *values, const V *default_value, const bool is_full_default,
      ThreadContext *thread_context,
      std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
          &reply,
      const int64_t begin, const int64_t max_i,
      const int64_t Velems_per_dim0) = 0;

  virtual Status MgetToTensorWithExist(
      V *values, const V *default_value, bool *exists,
      const bool is_full_default, ThreadContext *thread_context,
      std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
          &reply,
      const int64_t begin, const int64_t max_i,
      const int64_t Velems_per_dim0) = 0;

  virtual Status MsetCommand(
      const K *, const V *values, ThreadContext *thread_context,
      const int64_t begin, const int64_t max_i, const int64_t Velems_per_dim0,
      const std::vector<std::string> &keys_prefix_name_slices) = 0;

  virtual Status MaccumCommand(
      const K *, const V *values, const bool *exists,
      ThreadContext *thread_context, const int64_t begin, const int64_t max_i,
      const int64_t Velems_per_dim0, std::string &values_dtype_str,
      const std::vector<std::string> &keys_prefix_name_slices) = 0;

  virtual Status DelCommand(
      const K *, ThreadContext *thread_context, const int64_t begin,
      const int64_t max_i,
      const std::vector<std::string> &keys_prefix_name_slices) = 0;
};

template <typename RedisInstance, typename K, typename V, typename = void>
class RedisWrapper : public RedisBaseWrapper<K, V> {};

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

#if defined(__arm64__) || defined(__aarch64__)
#define CRC32CB(crc, value) \
  __asm__("crc32cb %w[c], %w[c], %w[v]" : [ c ] "+r"(crc) : [ v ] "r"(value))
#define CRC32CH(crc, value) \
  __asm__("crc32ch %w[c], %w[c], %w[v]" : [ c ] "+r"(crc) : [ v ] "r"(value))
#define CRC32CW(crc, value) \
  __asm__("crc32cw %w[c], %w[c], %w[v]" : [ c ] "+r"(crc) : [ v ] "r"(value))
#define CRC32CX(crc, value) \
  __asm__("crc32cx %w[c], %w[c], %x[v]" : [ c ] "+r"(crc) : [ v ] "r"(value))
#elif defined(__x86_64__)
#define CRC32CB(crc, value) __asm__("crc32b %1, %0" : "+r"(crc) : "rm"(value))
#define CRC32CH(crc, value) __asm__("crc32w %1, %0" : "+r"(crc) : "rm"(value))
#define CRC32CW(crc, value) __asm__("crc32l %1, %0" : "+r"(crc) : "rm"(value))
#define CRC32CX(crc, value) __asm__("crc32q %1, %0" : "+r"(crc) : "rm"(value))
#else
#error Currently architectures other than ARM64 or X86_64 are not supported
#endif

inline uint32_t crc32c_hash(uint32_t crc, const uint8_t *p, size_t length) {
#if defined(__arm64__) || defined(__aarch64__)
  int64_t len_int64_t = length;
  while ((len_int64_t -= sizeof(uint64_t)) >= 0) {
    CRC32CX(crc, *((uint64_t *)p));
    p += sizeof(uint64_t);
  }
  length &= (sizeof(uint64_t) - 1);
#elif defined(__x86_64__)
  int32_t len_int32 = length;
  while ((len_int32 -= sizeof(uint32_t)) >= 0) {
    CRC32CW(crc, *((uint32_t *)p));
    p += sizeof(uint32_t);
  }
  length &= (sizeof(uint32_t) - 1);
#else
#error Currently architectures other than ARM64 or X86_64 are not supported
#endif
  if (length & sizeof(uint32_t)) {
    CRC32CW(crc, *((uint32_t *)p));
    p += sizeof(uint32_t);
    length -= sizeof(uint32_t);
  }
  if (length & sizeof(uint16_t)) {
    CRC32CH(crc, *((uint16_t *)p));
    p += sizeof(uint16_t);
    length -= sizeof(uint16_t);
  }
  if (length & sizeof(uint8_t)) CRC32CB(crc, *p);
  return crc;
}

template <typename T,
          std::enable_if_t<(sizeof(T) <= 4) && !std::is_same<tstring, T>::value>
              * = nullptr>
unsigned KBucketNumCommonHandle(uint32_t crc, const uint8_t *p, size_t length) {
  return static_cast<const unsigned>(*reinterpret_cast<const T *>(p));
}

template <typename T,
          std::enable_if_t<(sizeof(T) > 4) && !std::is_same<tstring, T>::value>
              * = nullptr>
unsigned KBucketNumCommonHandle(uint32_t crc, const uint8_t *p, size_t length) {
  return static_cast<const unsigned>((*reinterpret_cast<const T *>(p)) &
                                     0x00000000FFFFFFFF);
}

template <typename T,
          std::enable_if_t<std::is_same<tstring, T>::value> * = nullptr>
unsigned KBucketNumCommonHandle(uint32_t crc, const uint8_t *p, size_t length) {
  return crc32c_hash(crc, p, length);
}

unsigned KBucketNumCRC32Handle(uint32_t crc, const uint8_t *p, size_t length) {
  return crc32c_hash(crc, p, length);
}

template <typename T>
inline unsigned KBucketNum(const KBucketNumHandle handle, const T *in,
                           const unsigned storage_slice) {
  assert(handle);
  const uint8_t *tem_char = reinterpret_cast<const uint8_t *>(in);
  return ((*handle)(0xffffffff, tem_char, sizeof(T))) % storage_slice;
}

template <>
inline unsigned KBucketNum<tstring>(const KBucketNumHandle handle,
                                    const tstring *in,
                                    const unsigned storage_slice) {
  const uint8_t *tem_char = reinterpret_cast<const uint8_t *>(in);
  return (crc32c_hash(0xffffffff, tem_char, in->length()) % storage_slice);
}

template <typename T>
inline const VContentAndTypeSizeResult &VContentAndTypeSize(
    VContentAndTypeSizeResult &_VContentAndTypeSizeResult,
    const int64_t Velems_per_dim0, const std::size_t &V_byte_size, const T *in,
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
    const int64_t Velems_per_dim0, const std::size_t &V_byte_size,
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
                           const int64_t Velems_per_dim0) {
  void *pv_raw_ = reinterpret_cast<void *>(const_cast<T *>(pv_raw));
  memcpy(pv_raw_, reinterpret_cast<const void *>(dft),
         Velems_per_dim0 *
             sizeof(T));  // Direct access to Tensor data in TensorFlow
}

template <>
void DefaultMemcpyToTensor<tstring>(const tstring *const pv_raw,
                                    const tstring *const dft,
                                    const int64_t Velems_per_dim0) {
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
         sizeof(T));  // Direct access to Tensor data in TensorFlow
}

template <>
void ReplyMemcpyToKeyTensor<tstring>(const tstring *const pk_raw,
                                     const char *str, const size_t &byte_size) {
  (const_cast<tstring *const>(pk_raw))->assign(str, byte_size);
}

template <typename T>
void ReplyMemcpyToValTensor(const T *const pv_raw, const char *str,
                            const int64_t Velems_per_dim0) {
  void *pv_raw_ = reinterpret_cast<void *>(const_cast<T *>(pv_raw));
  memcpy(pv_raw_, str,
         Velems_per_dim0 *
             sizeof(T));  // Direct access to Tensor data in TensorFlow
}

template <>
void ReplyMemcpyToValTensor<tstring>(const tstring *const pv_raw,
                                     const char *str,
                                     const int64_t Velems_per_dim0) {
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
