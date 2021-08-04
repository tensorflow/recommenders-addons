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

extern "C" {
#include <hiredis/sds.h>
}

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
  bool thread_context_i_status = false;

  for (; thread_context_id < threads_context.size(); ++thread_context_id) {
    thread_context_i_status = false;
    if (threads_context[thread_context_id]
            ->thread_occupied.compare_exchange_strong(
                thread_context_i_status, true, std::memory_order_seq_cst,
                std::memory_order_relaxed) == true) {
      break;
    }
  }
  if (thread_context_id == threads_context.size()) {
    std::lock_guard<std::mutex> guard(threads_context_mutex);
    threads_context.push_back(new ThreadContext());
    threads_context.back()->thread_occupied.store(true,
                                                  std::memory_order_release);
  }
  return thread_context_id;
}

void launchFindCore(std::shared_ptr<RedisVirtualWrapper> _table_instance,
                    std::vector<std::string> &keys_prefix_name_slices,
                    const Tensor &keys, Tensor *values,
                    const Tensor &default_value, const bool is_full_default,
                    const int64 &Velems_per_flat2_dim0,
                    std::vector<ThreadContext *> &threads_Find,
                    std::mutex &threads_Find_mutex, const int64 begin,
                    const int64 end) {
  size_t thread_context_id =
      SelectAvailableThreadContext(threads_Find, threads_Find_mutex);

  auto reply =
      _table_instance->MgetCommand(keys, threads_Find.at(thread_context_id),
                                   begin, end, keys_prefix_name_slices);

  _table_instance->MgetToTensor(values, default_value, is_full_default,
                                threads_Find.at(thread_context_id), reply,
                                begin, end, Velems_per_flat2_dim0);

  threads_Find[thread_context_id]->thread_occupied.store(
      false, std::memory_order_release);
}

void launchFindWithExistsCore(
    std::shared_ptr<RedisVirtualWrapper> _table_instance,
    std::vector<std::string> &keys_prefix_name_slices, const Tensor &keys,
    Tensor *values, const Tensor &default_value, Tensor &exists,
    const bool is_full_default, const int64 &Velems_per_flat2_dim0,
    std::vector<ThreadContext *> &threads_Find, std::mutex &threads_Find_mutex,
    const int64 begin, const int64 end) {
  // TODO: Implement the function of not looking up the table if the key does
  // not exist
  size_t thread_context_id =
      SelectAvailableThreadContext(threads_Find, threads_Find_mutex);

  auto reply =
      _table_instance->MgetCommand(keys, threads_Find.at(thread_context_id),
                                   begin, end, keys_prefix_name_slices);

  _table_instance->MgetToTensor(values, default_value, is_full_default,
                                threads_Find.at(thread_context_id), reply,
                                begin, end, Velems_per_flat2_dim0);

  threads_Find[thread_context_id]->thread_occupied.store(
      false, std::memory_order_release);
}

void launchInsertCore(std::shared_ptr<RedisVirtualWrapper> _table_instance,
                      std::vector<std::string> &keys_prefix_name_slices,
                      const Tensor &keys, const Tensor &values,
                      const int64 &Velems_per_flat2_dim0,
                      std::vector<ThreadContext *> &threads_Insert,
                      std::mutex &threads_Insert_mutex, const int64 begin,
                      const int64 end) {
  size_t thread_context_id =
      SelectAvailableThreadContext(threads_Insert, threads_Insert_mutex);

  _table_instance->MsetCommand(keys, values,
                               threads_Insert.at(thread_context_id), begin, end,
                               Velems_per_flat2_dim0, keys_prefix_name_slices);

  threads_Insert[thread_context_id]->thread_occupied.store(
      false, std::memory_order_release);
}

void launchDeleteCore(std::shared_ptr<RedisVirtualWrapper> _table_instance,
                      std::vector<std::string> &keys_prefix_name_slices,
                      const Tensor &keys,
                      std::vector<ThreadContext *> &threads_Delete,
                      std::mutex &threads_Delete_mutex, const int64 begin,
                      const int64 end) {
  size_t thread_context_id =
      SelectAvailableThreadContext(threads_Delete, threads_Delete_mutex);

  _table_instance->DelCommand(keys, threads_Delete.at(thread_context_id), begin,
                              end, keys_prefix_name_slices);

  threads_Delete[thread_context_id]->thread_occupied.store(
      false, std::memory_order_release);
}

void ParseJsonConfig(const std::string *const redis_config_abs_dir,
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
  }
  file_size = filestatus.st_size;
  file_contents = (char *)malloc(filestatus.st_size);
  if (file_contents == NULL) {
    LOG(ERROR) << "Memory error: unable to allocate "
               << std::to_string(file_size) << " bytes";
  }
  fp = fopen(filename, "rt");
  if (fp == NULL) {
    fclose(fp);
    free(file_contents);
    LOG(ERROR) << "Unable to open " << redis_config_abs_dir;
  }
  if (fread(file_contents, file_size, 1, fp) != 1) {
    fclose(fp);
    free(file_contents);
    LOG(ERROR) << "Unable t read content of " << redis_config_abs_dir;
  }
  fclose(fp);

  config_json = (json_char *)file_contents;
  config_value = json_parse(config_json, file_size);
  if (config_value->type != json_object) {
    free(file_contents);
    LOG(ERROR) << "Unable to parse the json data";
    throw(redis_config_abs_dir);
  }

  std::unordered_map<std::string, json_value *> json_hangar;
  std::unordered_map<std::string, json_value *>::iterator json_hangar_it;
  json_object_entry value_depth0_entry;
  json_value *value_depth1;
  for (unsigned i = 0; i < config_value->u.object.length; ++i) {
    value_depth0_entry = config_value->u.object.values[i];
    json_hangar[value_depth0_entry.name] = value_depth0_entry.value;
  }

  json_hangar_it = json_hangar.find("redis_connection_mode");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_integer) {
      redis_connection_params->redis_connection_mode =
          json_hangar_it->second->u.integer;
    }
  }

  json_hangar_it = json_hangar.find("redis_master_name");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_string) {
      redis_connection_params->redis_master_name =
          std::string(json_hangar_it->second->u.string.ptr,
                      json_hangar_it->second->u.string.length);
    }
  }

  json_hangar_it = json_hangar.find("redis_host_ip");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_array) {
      redis_connection_params->redis_host_ip.clear();
      for (unsigned i = 0; i < json_hangar_it->second->u.array.length; ++i) {
        value_depth1 = json_hangar_it->second->u.array.values[i];
        if (value_depth1->type == json_string) {
          redis_connection_params->redis_host_ip.push_back(std::string(
              value_depth1->u.string.ptr, value_depth1->u.string.length));
        }
      }
    }
  }

  json_hangar_it = json_hangar.find("redis_host_port");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_array) {
      redis_connection_params->redis_host_port.clear();
      for (unsigned i = 0; i < json_hangar_it->second->u.array.length; ++i) {
        value_depth1 = json_hangar_it->second->u.array.values[i];
        if (value_depth1->type == json_integer) {
          redis_connection_params->redis_host_port.push_back(
              value_depth1->u.integer);
        }
      }
    }
  }

  json_hangar_it = json_hangar.find("redis_password");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_string) {
      redis_connection_params->redis_password =
          std::string(json_hangar_it->second->u.string.ptr,
                      json_hangar_it->second->u.string.length);
    }
  }

  json_hangar_it = json_hangar.find("redis_db");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_integer) {
      redis_connection_params->redis_db = json_hangar_it->second->u.integer;
    }
  }

  json_hangar_it = json_hangar.find("redis_connect_timeout");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_integer) {
      redis_connection_params->redis_connect_timeout =
          json_hangar_it->second->u.integer;
    }
  }

  json_hangar_it = json_hangar.find("redis_socket_timeout");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_integer) {
      redis_connection_params->redis_socket_timeout =
          json_hangar_it->second->u.integer;
    }
  }

  json_hangar_it = json_hangar.find("redis_conn_pool_size");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_integer) {
      redis_connection_params->redis_conn_pool_size =
          json_hangar_it->second->u.integer;
    }
  }

  json_hangar_it = json_hangar.find("redis_wait_timeout");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_integer) {
      redis_connection_params->redis_wait_timeout =
          json_hangar_it->second->u.integer;
    }
  }

  json_hangar_it = json_hangar.find("redis_connection_lifetime");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_integer) {
      redis_connection_params->redis_connection_lifetime =
          json_hangar_it->second->u.integer;
    }
  }

  json_hangar_it = json_hangar.find("redis_sentinel_connect_timeout");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_integer) {
      redis_connection_params->redis_sentinel_connect_timeout =
          json_hangar_it->second->u.integer;
    }
  }

  json_hangar_it = json_hangar.find("redis_sentinel_socket_timeout");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_integer) {
      redis_connection_params->redis_sentinel_socket_timeout =
          json_hangar_it->second->u.integer;
    }
  }

  json_hangar_it = json_hangar.find("storage_slice");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_integer) {
      redis_connection_params->storage_slice_log2 =
          round_next_power_two_bitlen(json_hangar_it->second->u.integer);
      redis_connection_params->storage_slice =
          1 << redis_connection_params->storage_slice_log2;
      ;
    }
  }

  json_hangar_it = json_hangar.find("keys_sending_size");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_integer) {
      redis_connection_params->keys_sending_size =
          json_hangar_it->second->u.integer;
      ;
    }
  }

  json_hangar_it = json_hangar.find("using_md5_prefix_name");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_boolean) {
      redis_connection_params->using_md5_prefix_name =
          json_hangar_it->second->u.boolean;
    }
  }

  json_hangar_it = json_hangar.find("model_tag_old");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_string) {
      redis_connection_params->model_tag_old =
          std::string(json_hangar_it->second->u.string.ptr,
                      json_hangar_it->second->u.string.length);
    }
  }

  json_hangar_it = json_hangar.find("model_tag_new");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_string) {
      redis_connection_params->model_tag_new =
          std::string(json_hangar_it->second->u.string.ptr,
                      json_hangar_it->second->u.string.length);
    }
  }

  json_hangar_it = json_hangar.find("using_model_lib");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_boolean) {
      redis_connection_params->using_model_lib =
          json_hangar_it->second->u.boolean;
    }
  }

  json_hangar_it = json_hangar.find("model_lib_abs_dir");
  if (json_hangar_it != json_hangar.end()) {
    if (json_hangar_it->second->type == json_string) {
      redis_connection_params->model_lib_abs_dir =
          std::string(json_hangar_it->second->u.string.ptr,
                      json_hangar_it->second->u.string.length);
    }
  }

  json_value_free(config_value);
  free(file_contents);
}

std::string BuildKeysPrefixNameWithModelTag(const std::string &model_tag,
                                            const bool using_md5_prefix_name,
                                            const std::string &embedding_name) {
  std::array<unsigned char, 16> keys_prefix_name_md5;
  std::string keys_prefix_name;
  if (using_md5_prefix_name) {
    const std::string &&tmp_keys_prefix_name = model_tag + ":" + embedding_name;
    keys_prefix_name_md5 = MD5(tmp_keys_prefix_name);

    std::string md5_string;
    // char *md5_view_in_redis = sdscatrepr(
    //     sdsempty(), reinterpret_cast<char *>(keys_prefix_name_md5.data()),
    //     16);
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
              // << md5_view_in_redis
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
    const unsigned storage_slice, const std::string &keys_prefix_name) {
  std::vector<std::string> keys_prefix_name_slices;
  keys_prefix_name_slices.reserve(storage_slice);
  const unsigned bucket_num_in_redis = 16384 / storage_slice;
  for (unsigned i = 0; i < storage_slice; ++i) {
    keys_prefix_name_slices.push_back(
        keys_prefix_name + '{' +
        std::to_string(redis_slots_tab[bucket_num_in_redis * i]) + '}');
  }
  return keys_prefix_name_slices;
}

void CreateKeysPrefixNameHandle(
    const Redis_Connection_Params *const redis_connection_params,
    const std::string &embedding_name, std::string &keys_prefix_name_new,
    std::vector<std::string> &keys_prefix_name_slices_new,
    std::vector<std::string> &keys_prefix_name_slices_old) {
  keys_prefix_name_new = BuildKeysPrefixNameWithModelTag(
      redis_connection_params->model_tag_new,
      redis_connection_params->using_md5_prefix_name, embedding_name);
  keys_prefix_name_slices_new = BuildKeysPrefixNameSlices(
      redis_connection_params->storage_slice, keys_prefix_name_new);
  const std::string &keys_prefix_name_old = BuildKeysPrefixNameWithModelTag(
      redis_connection_params->model_tag_old,
      redis_connection_params->using_md5_prefix_name, embedding_name);
  keys_prefix_name_slices_old = BuildKeysPrefixNameSlices(
      redis_connection_params->storage_slice, keys_prefix_name_old);
}

}  // namespace redis_table
}  // namespace recommenders_addons
}  // namespace tensorflow
