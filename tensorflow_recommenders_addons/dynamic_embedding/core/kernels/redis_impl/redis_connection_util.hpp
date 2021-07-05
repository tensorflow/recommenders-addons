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
#pragma once
#include <aio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <x86intrin.h>

#include <cmath>
#include <iostream>

extern "C" {
#include <hiredis/sds.h>
}
#include "tensorflow/core/framework/types.h"
#include <sw/redis++/connection.h>
#include <sw/redis++/connection_pool.h>
#include <sw/redis++/redis++.h>
#include <sw/redis++/sentinel.h>

#include "md5.h"

namespace tensorflow {
namespace recommenders_addons {
namespace redis_connection {
/*
if v=0 or 1, return itself. if v<0, v=abs(v).
*/
inline unsigned round_next_power_two_bitlen(int v) {
  v = abs(v);
  if ((v | 1) == 1)
    return v;
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
    throw("The file " + path + " does not exist");
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
        if (ret == -1)
          return ret;
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
  if (access(path.c_str(), 0) == -1) // if folder doesn't exist
  {
    std::cout << "folder " << path << " doesn't exist" << std::endl;
    if (createDirectory(path) == 0) {
      std::cout << "folder " << path << " was created" << std::endl;
    } else {
      std::cout << "folder " << path << " failed to create" << std::endl;
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
  std::cerr << "Failed to parse the env-var ${" << env_var_name
            << "} into int32: " << _env_var_val
            << ". Use the default value: " << default_val << std::endl;
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
      1; // ClusterMode = 0, SentinelMode = 1, StreamMode = 2
  std::string redis_master_name = "master";
  // connection_options
  std::string redis_host_ip = "127.0.0.1";
  int redis_host_port = 26379;
  std::string redis_password = "";
  int redis_db = 0;
  //
  int redis_connect_timeout = 1000; // milliseconds
  int redis_socket_timeout = 1000;  // milliseconds
  // connection_pool_options
  int redis_conn_pool_size = 20;
  int redis_wait_timeout = 100000000;  // milliseconds
  int redis_connection_lifetime = 100; // minutes
  // sentinel_connection_options
  int redis_sentinel_connect_timeout = 1000; // milliseconds
  int sentinel_socket_timeout = 1000;        // milliseconds
  //
  // Below there is user-defined parameters in this custom op, not Redis
  // setting parameters
  unsigned storage_slice =
      1; // For deciding hash tag, which usually is how many Redis instance
         // may be used in the trainning.
  unsigned storage_slice_log2 = 0; // For fast calculation.
  std::string model_tag =
      "test"; //  model_tag for version and any other information
  bool using_MD5_prefix_name = false;
  std::string model_lib_abs_dir = "/tmp/";
  bool using_model_lib = true;

  Redis_Connection_Params &operator=(const Redis_Connection_Params &x) {
    redis_connection_mode = x.redis_connection_mode;
    redis_master_name = x.redis_master_name;
    redis_host_ip = x.redis_host_ip;
    redis_host_port = x.redis_host_port;
    redis_password = x.redis_password;
    redis_db = x.redis_db;
    redis_connect_timeout = x.redis_connect_timeout; // milliseconds
    redis_socket_timeout = x.redis_socket_timeout;   // milliseconds
    redis_conn_pool_size = x.redis_conn_pool_size;
    redis_wait_timeout = x.redis_wait_timeout;               // milliseconds
    redis_connection_lifetime = x.redis_connection_lifetime; // minutes
    redis_sentinel_connect_timeout =
        x.redis_sentinel_connect_timeout;                // milliseconds
    sentinel_socket_timeout = x.sentinel_socket_timeout; // milliseconds
    storage_slice_log2 =
        round_next_power_two_bitlen(x.storage_slice); // beter for modding.
    storage_slice = 1 << storage_slice_log2;
    model_tag = x.model_tag;
    using_MD5_prefix_name = x.using_MD5_prefix_name;
    model_lib_abs_dir = check_dir(x.model_lib_abs_dir);
    using_model_lib = x.using_model_lib;
    return *this;
  }
};

struct SlotContext {
  std::vector<const char *> ptrs;
  std::vector<std::size_t> sizes;
};

struct ThreadContext {
  std::vector<SlotContext> slots;
  std::vector<unsigned> slot_locs;

  void HandleReserve(const unsigned &storage_slice, const unsigned &vector_len,
                     const int &keys_num) {
    if (storage_slice > this->slots.size())
      this->slots.resize(storage_slice);
    for (unsigned i = 0; i < storage_slice; ++i) {
      this->slots[i].ptrs.clear();
      this->slots[i].ptrs.reserve(vector_len);
      this->slots[i].sizes.clear();
      this->slots[i].sizes.reserve(vector_len);
    }
    this->slot_locs.reserve(keys_num);
  }

  void HandlePushBack(const unsigned &slot_num, const char *ptrs_in,
                      const std::size_t &sizes_in) {
    this->slots[slot_num].ptrs.push_back(ptrs_in);
    this->slots[slot_num].sizes.push_back(sizes_in);
  }
};

class RedisVirtualWrapper {
protected:
  bool isRedisConnect = false;
  Redis_Connection_Params redis_connection_params;

public:
  void set_params(struct Redis_Connection_Params &conn_params_input) {
    this->redis_connection_params = conn_params_input;
  }

  virtual void conn() = 0;

  virtual bool check_slices_num(const std::string &keys_prefix_name) = 0;

  virtual size_t table_size_in_slots(
      const std::vector<std::string> &keys_prefix_name_slices) = 0;

  virtual void remove_hkeys_in_slots(
      const std::vector<std::string> &keys_prefix_name_slices) = 0;

  virtual std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
  get_keys_in_hkeys(
      const std::vector<std::string> &keys_prefix_name_slices) = 0;

  virtual void
  dump_to_disk(const std::vector<std::string> &keys_prefix_name_slices,
               std::vector<aiocb> &wrs, const std::vector<int> &fds) = 0;

  virtual void
  restore_from_disk(const std::vector<std::string> &keys_prefix_name_slices,
                    std::vector<aiocb> &rds, const std::vector<int> &fds,
                    const std::vector<unsigned long> &buf_sizes) = 0;

  virtual std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
  MGET_COMMAND(const ::tensorflow::Tensor &keys, ThreadContext &thread_context,
               const ::tensorflow::int64 &begin,
               const ::tensorflow::int64 &max_i,
               const std::vector<std::string> &keys_prefix_name_slices) = 0;

  virtual void MGET_to_Tensor(
      ::tensorflow::Tensor *values, const ::tensorflow::Tensor &default_value,
      const bool &is_full_default, ThreadContext &thread_context,
      std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
          &reply,
      const ::tensorflow::int64 &begin, const ::tensorflow::int64 &max_i) = 0;

  virtual void
  MSET_COMMAND(const ::tensorflow::Tensor &keys,
               const ::tensorflow::Tensor &values,
               ThreadContext &thread_context, const ::tensorflow::int64 &begin,
               const ::tensorflow::int64 &max_i,
               const std::vector<std::string> &keys_prefix_name_slices) = 0;

  virtual void
  DEL_COMMAND(const ::tensorflow::Tensor &keys, ThreadContext &thread_context,
              const ::tensorflow::int64 &begin,
              const ::tensorflow::int64 &max_i,
              const std::vector<std::string> &keys_prefix_name_slices) = 0;
};

template <typename RedisInstance, typename K, typename V, typename = void>
class RedisWrapper : public RedisVirtualWrapper {};

struct VContentAndTypeSizeResult {
  size_t VTypeSize;
  const char *VContentPointer;
};

template <typename T> constexpr inline size_t KTypeSize(const T *in) {
  return sizeof(T);
}

template <>
inline size_t
KTypeSize<::tensorflow::tstring>(const ::tensorflow::tstring *in) {
  return in->size();
}

template <typename T>
constexpr inline const char *KContentPointer(const T *in) {
  return reinterpret_cast<const char *>(in);
}

template <>
inline const char *
KContentPointer<::tensorflow::tstring>(const ::tensorflow::tstring *in) {
  return in->data();
}

template <typename T>
inline unsigned KSlotNum(const T *in, const unsigned storage_slice) {
  return static_cast<const int>(*in) & (storage_slice - 1);
}

template <>
inline unsigned KSlotNum<::tensorflow::tstring>(const ::tensorflow::tstring *in,
                                                const unsigned storage_slice) {
  const auto tem_char = *(reinterpret_cast<const unsigned char *>(in));
  return (_mm_crc32_u8(0xffffffff, tem_char) & (storage_slice - 1));
}

template <typename T>
inline const VContentAndTypeSizeResult &
VContentAndTypeSize(VContentAndTypeSizeResult &_VContentAndTypeSizeResult,
                    const ::tensorflow::int64 &Velems_per_dim0,
                    const std::size_t &V_byte_size, const T *in,
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
[str_size,string[...],str_size,string[...],str_size,string[...],...]
*/
template <>
inline const VContentAndTypeSizeResult &
VContentAndTypeSize<::tensorflow::tstring>(
    VContentAndTypeSizeResult &_VContentAndTypeSizeResult,
    const ::tensorflow::int64 &Velems_per_dim0, const std::size_t &V_byte_size,
    const ::tensorflow::tstring *in, std::vector<char> &buff) {
  const ::tensorflow::tstring *ps_end = in + Velems_per_dim0;

  _VContentAndTypeSizeResult.VContentPointer = &buff.back();

  size_t tot = 0;

  for (const ::tensorflow::tstring *ps = in; ps != ps_end; ++ps) {
    tot = tot + ps->size() + sizeof(int);
  }

  _VContentAndTypeSizeResult.VTypeSize = tot;
  buff.reserve(buff.size() + tot);

  int tem_byte_size = 0;
  for (const ::tensorflow::tstring *ps = in; ps != ps_end; ++ps) {
    tem_byte_size = static_cast<int>(ps->size());
    buff.insert(buff.end(), &tem_byte_size,
                (&tem_byte_size) +
                    sizeof(int)); // Firstly store how long the string is
    buff.insert(buff.end(), ps->begin(), ps->end());
  }
  return _VContentAndTypeSizeResult;
}

template <typename T>
void default_memcpy_to_tensor(T *pv_raw, const T *dft,
                              const ::tensorflow::int64 &Velems_per_dim0) {
  memcpy(reinterpret_cast<void *>(pv_raw), reinterpret_cast<const void *>(dft),
         Velems_per_dim0 *
             sizeof(T)); // Direct access to Tensor data in TensorFlow
}

template <>
void default_memcpy_to_tensor<::tensorflow::tstring>(
    ::tensorflow::tstring *const pv_raw, const ::tensorflow::tstring *const dft,
    const ::tensorflow::int64 &Velems_per_dim0) {
  const ::tensorflow::tstring *const pv_raw_end = pv_raw + Velems_per_dim0;
  ::tensorflow::tstring *pv_it = pv_raw;
  const ::tensorflow::tstring *dft_it = dft;
  for (; pv_it != pv_raw_end; ++pv_it, ++dft_it) {
    *pv_it = *dft_it;
  }
}

template <typename T>
void reply_memcpy_to_tensor(T *pv_raw, const char *str,
                            const ::tensorflow::int64 &Velems_per_dim0) {
  memcpy(reinterpret_cast<void *>(pv_raw), str,
         Velems_per_dim0 *
             sizeof(T)); // Direct access to Tensor data in TensorFlow
}

template <>
void reply_memcpy_to_tensor<::tensorflow::tstring>(
    ::tensorflow::tstring *const pv_raw, const char *str,
    const ::tensorflow::int64 &Velems_per_dim0) {
  const ::tensorflow::tstring *const pv_raw_end = pv_raw + Velems_per_dim0;
  const char *char_view = str;
  int str_bytesize = 0;
  for (::tensorflow::tstring *pv_it = pv_raw; pv_it != pv_raw_end; ++pv_it) {
    str_bytesize = *(reinterpret_cast<const int *>(char_view));
    char_view += sizeof(int);
    pv_it->assign(char_view, str_bytesize);
    char_view += str_bytesize;
  }
}

} // namespace redis_connection
} // namespace recommenders_addons
} // namespace tensorflow
