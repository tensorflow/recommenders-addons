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

#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include <type_traits>
#include <utility>

#include "json.h"
#include "redis_connection_util.hpp"
#include "redis_slots_tab.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/types.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/utils.h"

namespace tensorflow {
namespace recommenders_addons {
namespace redis_table {

using namespace redis_connection;

size_t SelectAvailableThreadContext(
    std::vector<ThreadContext *> &threads_context,
    std::mutex &threads_context_mutex) {
  size_t thread_context_id = 0;
  bool *thread_context_i_status = new bool[threads_context.size()]();

  for (; thread_context_id < threads_context.size(); ++thread_context_id) {
    thread_context_i_status[thread_context_id] = false;
    if (threads_context[thread_context_id]
            ->thread_occupied.compare_exchange_strong(
                thread_context_i_status[thread_context_id], true,
                std::memory_order_seq_cst) == true) {
      break;
    }
  }
  if (thread_context_id == threads_context.size()) {
    threads_context.push_back(new ThreadContext());
    threads_context.back()->thread_occupied.store(true,
                                                  std::memory_order_release);
  }
  delete[] thread_context_i_status;
  return thread_context_id;
}

template <typename K, typename V>
Status launchFindCore(std::shared_ptr<RedisBaseWrapper<K, V>> _table_instance,
                      std::vector<std::string> &keys_prefix_name_slices,
                      const K *keys, V *values, const V *default_value,
                      const bool is_full_default,
                      const int64_t &Velems_per_flat2_dim0,
                      std::vector<ThreadContext *> &threads_Find,
                      std::mutex &threads_Find_mutex, const int64_t begin,
                      const int64_t end) {
  size_t thread_context_id =
      SelectAvailableThreadContext(threads_Find, threads_Find_mutex);

  auto reply =
      _table_instance->MgetCommand(keys, threads_Find.at(thread_context_id),
                                   begin, end, keys_prefix_name_slices);

  auto statu =
      _table_instance->MgetToTensor(values, default_value, is_full_default,
                                    threads_Find.at(thread_context_id), reply,
                                    begin, end, Velems_per_flat2_dim0);

  threads_Find[thread_context_id]->thread_occupied.store(
      false, std::memory_order_release);

  return statu;
}

template <typename K, typename V>
Status launchFindWithExistsCore(
    std::shared_ptr<RedisBaseWrapper<K, V>> _table_instance,
    std::vector<std::string> &keys_prefix_name_slices, const K *keys, V *values,
    const V *default_value, bool *exists, const bool is_full_default,
    const int64_t &Velems_per_flat2_dim0,
    std::vector<ThreadContext *> &threads_Find, std::mutex &threads_Find_mutex,
    const int64_t begin, const int64_t end) {
  // TODO: Implement the function of not looking up the table if the key does
  // not exist
  size_t thread_context_id =
      SelectAvailableThreadContext(threads_Find, threads_Find_mutex);

  auto reply =
      _table_instance->MgetCommand(keys, threads_Find.at(thread_context_id),
                                   begin, end, keys_prefix_name_slices);

  auto statu = _table_instance->MgetToTensorWithExist(
      values, default_value, exists, is_full_default,
      threads_Find.at(thread_context_id), reply, begin, end,
      Velems_per_flat2_dim0);

  threads_Find[thread_context_id]->thread_occupied.store(
      false, std::memory_order_release);

  return statu;
}

template <typename K, typename V>
Status launchInsertCore(std::shared_ptr<RedisBaseWrapper<K, V>> _table_instance,
                        std::vector<std::string> &keys_prefix_name_slices,
                        const K *keys, const V *values,
                        const int64_t &Velems_per_flat2_dim0,
                        std::vector<ThreadContext *> &threads_Insert,
                        std::mutex &threads_Insert_mutex, const int64_t begin,
                        const int64_t end) {
  size_t thread_context_id =
      SelectAvailableThreadContext(threads_Insert, threads_Insert_mutex);

  auto statu = _table_instance->MsetCommand(
      keys, values, threads_Insert.at(thread_context_id), begin, end,
      Velems_per_flat2_dim0, keys_prefix_name_slices);

  threads_Insert[thread_context_id]->thread_occupied.store(
      false, std::memory_order_release);

  return statu;
}

template <typename K, typename V>
Status launchAccumCore(std::shared_ptr<RedisBaseWrapper<K, V>> _table_instance,
                       std::vector<std::string> &keys_prefix_name_slices,
                       const K *keys, const V *values_or_delta,
                       const bool *exists, const int64_t &Velems_per_flat2_dim0,
                       std::string &values_dtype_str,
                       std::vector<ThreadContext *> &threads_Insert,
                       std::mutex &threads_Accum_mutex, const int64_t begin,
                       const int64_t end) {
  size_t thread_context_id =
      SelectAvailableThreadContext(threads_Insert, threads_Accum_mutex);

  auto statu = _table_instance->MaccumCommand(
      keys, values_or_delta, exists, threads_Insert.at(thread_context_id),
      begin, end, Velems_per_flat2_dim0, values_dtype_str,
      keys_prefix_name_slices);

  threads_Insert[thread_context_id]->thread_occupied.store(
      false, std::memory_order_release);

  return statu;
}

template <typename K, typename V>
Status launchDeleteCore(std::shared_ptr<RedisBaseWrapper<K, V>> _table_instance,
                        std::vector<std::string> &keys_prefix_name_slices,
                        const K *keys,
                        std::vector<ThreadContext *> &threads_Delete,
                        std::mutex &threads_Delete_mutex, const int64_t begin,
                        const int64_t end) {
  size_t thread_context_id =
      SelectAvailableThreadContext(threads_Delete, threads_Delete_mutex);

  auto statu =
      _table_instance->DelCommand(keys, threads_Delete.at(thread_context_id),
                                  begin, end, keys_prefix_name_slices);

  threads_Delete[thread_context_id]->thread_occupied.store(
      false, std::memory_order_release);

  return statu;
}

Status ParseJsonConfig(const std::string *const redis_config_abs_dir,
                       Redis_Connection_Params *redis_connection_params) {
  const char *filename = redis_config_abs_dir->c_str();
  FILE *fp;
  struct stat filestatus;
  size_t file_size;
  char *file_contents;
  json_char *config_json;
  json_value *config_value;

  if (stat(filename, &filestatus) != 0) {
    LOG(ERROR) << "File " << filename << " not found";
    return errors::NotFound("File ", filename, " not found");
  }
  file_size = filestatus.st_size;
  file_contents = (char *)malloc(file_size);
  if (file_contents == NULL) {
    LOG(ERROR) << "Memory error: unable to allocate "
               << std::to_string(file_size) << " bytes";
    return errors::ResourceExhausted("Memory error: unable to allocate ",
                                     std::to_string(file_size), " bytes");
  }
  fp = fopen(filename, "rt");
  if (fp == NULL) {
    fclose(fp);
    free(file_contents);
    LOG(ERROR) << "Unable to open " << redis_config_abs_dir;
    return errors::PermissionDenied("Unable to open ", redis_config_abs_dir);
  }
  if (fread(file_contents, file_size, 1, fp) != 1) {
    fclose(fp);
    free(file_contents);
    LOG(ERROR) << "Unable t read content of " << redis_config_abs_dir;
    return errors::Unavailable("Unable t read content of ",
                               redis_config_abs_dir);
  }
  fclose(fp);

  config_json = (json_char *)file_contents;
  config_value = json_parse(config_json, file_size);
  if (config_value->type != json_object) {
    free(file_contents);
    LOG(ERROR) << "Unable to parse the json data";
    return errors::Unknown("Unable to parse the json data from ",
                           redis_config_abs_dir);
  }

  std::unordered_map<std::string, json_value *> json_hangar;
  std::unordered_map<std::string, json_value *>::iterator json_hangar_it;
  json_object_entry value_depth0_entry;
  json_value *value_depth1;
  for (unsigned i = 0; i < config_value->u.object.length; ++i) {
    value_depth0_entry = config_value->u.object.values[i];
    json_hangar[value_depth0_entry.name] = value_depth0_entry.value;
  }

#define ReadOneJsonToParams(json_key_name, json_val_type)                \
  {                                                                      \
    json_hangar_it = json_hangar.find(#json_key_name);                   \
    if (json_hangar_it != json_hangar.end()) {                           \
      if (json_hangar_it->second->type == json_##json_val_type) {        \
        redis_connection_params->json_key_name =                         \
            json_hangar_it->second->u.json_val_type;                     \
      } else {                                                           \
        LOG(ERROR) << #json_key_name " should be json " #json_val_type;  \
        return Status(error::INVALID_ARGUMENT,                           \
                      #json_key_name " should be json " #json_val_type); \
      }                                                                  \
    }                                                                    \
  }

#define ReadStringOneJsonToParams(json_key_name)                  \
  {                                                               \
    json_hangar_it = json_hangar.find(#json_key_name);            \
    if (json_hangar_it != json_hangar.end()) {                    \
      if (json_hangar_it->second->type == json_string) {          \
        redis_connection_params->json_key_name =                  \
            std::string(json_hangar_it->second->u.string.ptr,     \
                        json_hangar_it->second->u.string.length); \
      } else {                                                    \
        LOG(ERROR) << #json_key_name " should be json string";    \
        return Status(error::INVALID_ARGUMENT,                    \
                      #json_key_name " should be json string");   \
      }                                                           \
    }                                                             \
  }

#define ReadArrayJsonToParams(json_key_name, json_val_type)                \
  {                                                                        \
    json_hangar_it = json_hangar.find(#json_key_name);                     \
    if (json_hangar_it != json_hangar.end()) {                             \
      if (json_hangar_it->second->type == json_array) {                    \
        redis_connection_params->json_key_name.clear();                    \
        for (unsigned i = 0; i < json_hangar_it->second->u.array.length;   \
             ++i) {                                                        \
          value_depth1 = json_hangar_it->second->u.array.values[i];        \
          if (value_depth1->type == json_##json_val_type) {                \
            redis_connection_params->redis_host_port.push_back(            \
                value_depth1->u.json_val_type);                            \
          } else {                                                         \
            LOG(ERROR) << #json_key_name " should be json " #json_val_type \
                                         " array";                         \
            return Status(error::INVALID_ARGUMENT, #json_key_name          \
                          " should be json " #json_val_type " array");     \
          }                                                                \
        }                                                                  \
      } else {                                                             \
        LOG(ERROR) << #json_key_name " should be json " #json_val_type     \
                                     " array";                             \
        return Status(error::INVALID_ARGUMENT, #json_key_name              \
                      " should be json " #json_val_type " array");         \
      }                                                                    \
    }                                                                      \
  }

#define ReadStringArrayJsonToParams(json_key_name)                           \
  {                                                                          \
    json_hangar_it = json_hangar.find(#json_key_name);                       \
    if (json_hangar_it != json_hangar.end()) {                               \
      if (json_hangar_it->second->type == json_array) {                      \
        redis_connection_params->json_key_name.clear();                      \
        for (unsigned i = 0; i < json_hangar_it->second->u.array.length;     \
             ++i) {                                                          \
          value_depth1 = json_hangar_it->second->u.array.values[i];          \
          if (value_depth1->type == json_string) {                           \
            redis_connection_params->json_key_name.push_back(std::string(    \
                value_depth1->u.string.ptr, value_depth1->u.string.length)); \
          } else {                                                           \
            LOG(ERROR) << #json_key_name " should be json string array";     \
            return Status(error::INVALID_ARGUMENT,                           \
                          #json_key_name " should be json string array");    \
          }                                                                  \
        }                                                                    \
      } else {                                                               \
        LOG(ERROR) << #json_key_name " should be json string array";         \
        return Status(error::INVALID_ARGUMENT,                               \
                      #json_key_name " should be json string array");        \
      }                                                                      \
    }                                                                        \
  }

  ReadOneJsonToParams(redis_connection_mode, integer);

  ReadStringOneJsonToParams(redis_master_name);

  ReadStringArrayJsonToParams(redis_host_ip);

  ReadArrayJsonToParams(redis_host_port, integer);

  ReadStringOneJsonToParams(redis_user);

  ReadStringOneJsonToParams(redis_password);

  ReadOneJsonToParams(redis_db, integer);

  ReadOneJsonToParams(redis_read_access_slave, boolean);

  ReadOneJsonToParams(redis_connect_keep_alive, boolean);

  ReadOneJsonToParams(redis_connect_timeout, integer);

  ReadOneJsonToParams(redis_socket_timeout, integer);

  ReadOneJsonToParams(redis_conn_pool_size, integer);

  ReadOneJsonToParams(redis_wait_timeout, integer);

  ReadOneJsonToParams(redis_connection_lifetime, integer);

  ReadStringOneJsonToParams(redis_sentinel_user);

  ReadStringOneJsonToParams(redis_sentinel_password);

  ReadOneJsonToParams(redis_sentinel_connect_timeout, integer);

  ReadOneJsonToParams(redis_sentinel_socket_timeout, integer);

  ReadOneJsonToParams(storage_slice_import, integer);

  ReadOneJsonToParams(storage_slice, integer);

  redis_connection_params->storage_slice_import =
      redis_connection_params->storage_slice_import > 0
          ? redis_connection_params->storage_slice_import
          : redis_connection_params->storage_slice;

  ReadOneJsonToParams(using_hash_storage_slice, boolean);

  ReadOneJsonToParams(keys_sending_size, integer);

  ReadOneJsonToParams(using_md5_prefix_name, boolean);

  ReadOneJsonToParams(redis_hash_tags_hypodispersion, boolean);

  ReadStringOneJsonToParams(model_tag_import);

  ReadStringArrayJsonToParams(redis_hash_tags_import);

  ReadStringOneJsonToParams(model_tag_runtime);

  ReadStringArrayJsonToParams(redis_hash_tags_runtime);

  ReadOneJsonToParams(expire_model_tag_in_seconds, integer);

  ReadOneJsonToParams(table_store_mode, integer);

  ReadStringOneJsonToParams(model_lib_abs_dir);

#undef ReadOneJsonToParams
#undef ReadStringOneJsonToParams
#undef ReadArrayJsonToParams
#undef ReadStringArrayJsonToParams

  json_value_free(config_value);
  free(file_contents);

  return TFOkStatus;
}

extern "C" sds sdsempty(void);
extern "C" sds sdscatrepr(sds s, const char *p, size_t len);
std::string BuildKeysPrefixNameWithModelTag(const std::string &model_tag,
                                            const bool using_md5_prefix_name,
                                            const std::string &embedding_name) {
  std::array<unsigned char, 16> keys_prefix_name_md5;
  std::string keys_prefix_name;
  if (using_md5_prefix_name) {
    const std::string &&tmp_keys_prefix_name = model_tag + ":" + embedding_name;
    keys_prefix_name_md5 = MD5(tmp_keys_prefix_name);

    std::string md5_string;
    char *md5_view_in_redis = sdscatrepr(
        sdsempty(), reinterpret_cast<char *>(keys_prefix_name_md5.data()), 16);
    char tmp[3];
    for (int i = 0; i < 16; ++i) {
      memset(tmp, 0x00, sizeof(tmp));
      sprintf(tmp, "%02X", keys_prefix_name_md5[i]);
      md5_string += tmp;
    }
    LOG(INFO) << "Init table tensor, now prefix name for keys namespace is "
              << keys_prefix_name << ". The MD5 of prefix name for keys is "
              << md5_string
              << ". And Its characters view in redis namespace is "
              << md5_view_in_redis
              << ". This MD5 is used to store keys for distinguishing "
                 "between different model and table names";

    keys_prefix_name =
        std::string(reinterpret_cast<char *>(keys_prefix_name_md5.data()), 16);
  } else {
    keys_prefix_name = model_tag + ":" + embedding_name;
  }

  return keys_prefix_name;
}

std::vector<std::string> BuildKeysPrefixNameSlices(
    const std::vector<std::pair<unsigned, unsigned>> cluster_slots,
    const unsigned storage_slice, std::vector<std::string> redis_hash_tags,
    const std::string &keys_prefix_name) {
  std::vector<std::string> keys_prefix_name_slices;
  keys_prefix_name_slices.reserve(storage_slice);
  if (redis_hash_tags.size() == storage_slice) {
    LOG(INFO) << "Using the prefix redis_hash_tags for every bucket.";
    for (auto redis_hash_tag : redis_hash_tags) {
      if (redis_hash_tag.back() != '}') {
        redis_hash_tag.push_back('}');
      }
      if (redis_hash_tag.front() != '{') {
        redis_hash_tag.insert(redis_hash_tag.begin(), '{');
      }
      keys_prefix_name_slices.emplace_back(keys_prefix_name + redis_hash_tag);
    }
  } else {
    LOG(INFO)
        << "Number of prefix redis_hash_tags is not equal to the prefix "
           "storage_slice. Now using the hash tags generated sequentially.";
    if (cluster_slots.size() == 0) {
      const unsigned slot_num_in_redis = 16384 / storage_slice;
      const unsigned slot_in_redis_rem = 16384 % storage_slice;
      for (unsigned i = 0; i < storage_slice; ++i) {
        keys_prefix_name_slices.emplace_back(
            keys_prefix_name + '{' +
            std::to_string(
                redis_slots_tab[slot_num_in_redis * i + slot_in_redis_rem]) +
            '}');
      }
    } else {
      if (cluster_slots.size() < storage_slice) {
        for (unsigned i = 0; i < storage_slice;) {
          for (auto cluster_slot : cluster_slots) {
            keys_prefix_name_slices.emplace_back(
                keys_prefix_name + '{' +
                std::to_string(redis_slots_tab[cluster_slot.first +
                                               i / cluster_slots.size() *
                                                   ((cluster_slot.second -
                                                     cluster_slot.first) *
                                                    cluster_slots.size() /
                                                    storage_slice)]) +
                '}');
            ++i;
            if (i == storage_slice) {
              break;
            }
          }
        }
      } else {
        if (cluster_slots.size() > storage_slice) {
          LOG(WARNING) << "Nodes in Redis service is bigger than storage_slice "
                          "set by user, it may cause data skew.";
        }
        for (unsigned i = 0; i < storage_slice; ++i) {
          keys_prefix_name_slices.emplace_back(
              keys_prefix_name + '{' +
              std::to_string(redis_slots_tab[cluster_slots.at(i).first]) + '}');
        }
      }
    }
  }
  return keys_prefix_name_slices;
}

void CreateKeysPrefixNameHandle(
    const std::vector<std::pair<unsigned, unsigned>> cluster_slots,
    const Redis_Connection_Params *const redis_connection_params,
    const std::string &embedding_name, std::string &keys_prefix_name_runtime,
    std::string &keys_prefix_name_import,
    std::vector<std::string> &keys_prefix_name_slices_runtime,
    std::vector<std::string> &keys_prefix_name_slices_import) {
  keys_prefix_name_runtime = BuildKeysPrefixNameWithModelTag(
      redis_connection_params->model_tag_runtime,
      redis_connection_params->using_md5_prefix_name, embedding_name);
  keys_prefix_name_slices_runtime = BuildKeysPrefixNameSlices(
      cluster_slots, redis_connection_params->storage_slice,
      redis_connection_params->redis_hash_tags_runtime,
      keys_prefix_name_runtime);
  keys_prefix_name_import = BuildKeysPrefixNameWithModelTag(
      redis_connection_params->model_tag_import,
      redis_connection_params->using_md5_prefix_name, embedding_name);
  keys_prefix_name_slices_import = BuildKeysPrefixNameSlices(
      cluster_slots,
      static_cast<unsigned>(redis_connection_params->storage_slice_import),
      redis_connection_params->redis_hash_tags_import, keys_prefix_name_import);
}

}  // namespace redis_table
}  // namespace recommenders_addons
}  // namespace tensorflow
