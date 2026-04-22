#ifndef TFRA_CORE_LIB_PARLAYHASH_CONCURRENT_HASH_MAP_H_
#define TFRA_CORE_LIB_PARLAYHASH_CONCURRENT_HASH_MAP_H_

#include <cstddef>
#include <cstring>
#include <functional>
#include <utility>

#include "parlay_hash/unordered_map.h"

namespace tensorflow {
namespace recommenders_addons {
namespace concurrent {

template <class Key, class T, class Hash = parlay::parlay_hash<Key>>
class ConcurrentFlatHashMap {
public:
  using Map = parlay::parlay_unordered_map<Key, T, Hash>;

  explicit ConcurrentFlatHashMap(size_t init_size = 0)
      : map_(std::max<size_t>(init_size, 16)) {}

  ~ConcurrentFlatHashMap() = default;

  bool find(const Key &key, T &val) const {
    auto result = map_.Find(key);
    if (result.has_value()) {
      val = *result;
      return true;
    }
    return false;
  }

  template <typename F> bool find_fn(const Key &key, F fn) const {
    auto result = map_.Find(key);
    if (result.has_value()) {
      fn(*result);
      return true;
    }
    return false;
  }

  template <typename V> bool insert_or_assign(Key &&key, V &&val) {
    map_.Upsert(std::forward<Key>(key),
                [&](std::optional<T> v) { return std::forward<V>(val); });
    return true;
  }

  template <typename V> bool insert_or_assign(const Key &key, V &&val) {
    map_.Upsert(key, [&](std::optional<T> v) { return std::forward<V>(val); });
    return true;
  }

  template <typename V>
  bool insert_or_accum(Key &&key, V &&val_or_delta, bool exist) {
    if (exist) {
      map_.Upsert(std::forward<Key>(key), [&](std::optional<T> v) {
        if (v.has_value()) {
          auto tmp = *v;
          tmp += val_or_delta;
          return tmp;
        }
        return std::forward<V>(val_or_delta);
      });
    } else {
      map_.Insert(std::forward<Key>(key), std::forward<V>(val_or_delta));
    }
    return true;
  }

  template <typename V>
  bool insert_or_accum(const Key &key, V &&val_or_delta, bool exist) {
    if (exist) {
      map_.Upsert(key, [&](std::optional<T> v) {
        if (v.has_value()) {
          auto tmp = *v;
          tmp += val_or_delta;
          return tmp;
        }
        return std::forward<V>(val_or_delta);
      });
    } else {
      map_.Insert(key, std::forward<V>(val_or_delta));
    }
    return true;
  }

  bool erase(const Key &key) { return map_.Remove(key).has_value(); }

  void clear() { map_.clear(); }

  size_t size() const { return map_.size(); }

  bool empty() const { return map_.size() == 0; }

  template <typename F> void for_each(F fn) const {
    for (const auto &entry : map_) {
      fn(entry);
    }
  }

  void prefetch(const Key &) const {}

  size_t dump(Key *keys, T *values, const size_t search_offset,
              const size_t search_length) const {
    size_t count = 0;
    size_t skipped = 0;
    for (const auto &entry : map_) {
      if (skipped < search_offset) {
        ++skipped;
        continue;
      }
      if (count >= search_length)
        break;
      keys[count] = entry.first;
      values[count] = entry.second;
      ++count;
    }
    return count;
  }

private:
  mutable Map map_;
};

} // namespace concurrent
} // namespace recommenders_addons
} // namespace tensorflow

#endif // TFRA_CORE_LIB_PARLAYHASH_CONCURRENT_HASH_MAP_H_
