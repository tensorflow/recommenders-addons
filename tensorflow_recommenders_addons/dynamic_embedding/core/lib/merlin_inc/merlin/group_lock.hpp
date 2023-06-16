/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http:///www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Implementing a group mutex and relative lock guard for better E2E performance:
 * - Allow multiple writers (like `insert_or_assign` `assign` `insert_and_evict` etc.)
 *   The CUDA kernels guarantee the data consistency in this situation.
 * - Allow multiple readers (like `find` 'size` etc.)
 * - Not allow readers and writers to run concurrently
 * - The `write_read_lock` is used for special APIs (like `reserve` `erase` `clear` etc.)
 */
#include <atomic>
#include <cassert>
#include <mutex>
#include <system_error>
#include <thread>

namespace nv {
namespace merlin {

class group_shared_mutex {
 public:
  group_shared_mutex(const group_shared_mutex&) = delete;
  group_shared_mutex& operator=(const group_shared_mutex&) = delete;

  group_shared_mutex() noexcept
      : writer_count_(0), reader_count_(0), unique_flag_(false) {}

  void lock_read() {
    for (;;) {
      while (writer_count_.load(std::memory_order_acquire)) {
      }
      reader_count_.fetch_add(1, std::memory_order_acq_rel);
      if (writer_count_.load(std::memory_order_acquire) == 0) {
        break;
      }
      reader_count_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  void unlock_read() { reader_count_.fetch_sub(1, std::memory_order_release); }

  void lock_write() {
    for (;;) {
      while (reader_count_.load(std::memory_order_acquire)) {
      }
      writer_count_.fetch_add(1, std::memory_order_acq_rel);
      if (reader_count_.load(std::memory_order_acquire) == 0) {
        break;
      }
      writer_count_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  void unlock_write() { writer_count_.fetch_sub(1, std::memory_order_release); }

  void lock_write_read() {
    /* Lock unique flag */
    bool expected = false;
    while (!unique_flag_.compare_exchange_weak(expected, true,
                                               std::memory_order_acq_rel)) {
      expected = false;
    }

    /* Ban writer */
    for (;;) {
      while (writer_count_.load(std::memory_order_acquire)) {
      }
      reader_count_.fetch_add(1, std::memory_order_acq_rel);
      if (writer_count_.load(std::memory_order_acquire) == 0) {
        break;
      }
      reader_count_.fetch_sub(1, std::memory_order_acq_rel);
    }

    /* Ban reader */
    for (;;) {
      while (reader_count_.load(std::memory_order_acquire) > 1) {
      }
      writer_count_.fetch_add(1, std::memory_order_acq_rel);
      if (reader_count_.load(std::memory_order_acquire) == 1) {
        break;
      }
      writer_count_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  void unlock_write_read() noexcept {
    reader_count_.fetch_sub(1, std::memory_order_release);
    writer_count_.fetch_sub(1, std::memory_order_release);
    unique_flag_.store(false, std::memory_order_release);
  }

  int writer_count() noexcept {
    return writer_count_.load(std::memory_order_relaxed);
  }

  int reader_count() noexcept {
    return reader_count_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<int> writer_count_;
  std::atomic<int> reader_count_;
  std::atomic<bool> unique_flag_;
};

class reader_shared_lock {
 public:
  reader_shared_lock(const reader_shared_lock&) = delete;
  reader_shared_lock(reader_shared_lock&&) = delete;

  reader_shared_lock& operator=(const reader_shared_lock&) = delete;
  reader_shared_lock& operator=(reader_shared_lock&&) = delete;

  explicit reader_shared_lock(group_shared_mutex& mutex) : mutex_(&mutex) {
    mutex_->lock_read();
    owns_ = true;
  }

  explicit reader_shared_lock(group_shared_mutex& mutex, std::defer_lock_t)
      : mutex_(&mutex), owns_(false) {}

  ~reader_shared_lock() {
    if (owns_) {
      mutex_->unlock_read();
    }
  }

  void lock() noexcept {
    if (!owns_) {
      mutex_->lock_read();
      owns_ = true;
    }
  }

  bool owns_lock() const noexcept { return owns_; }

 private:
  group_shared_mutex* const mutex_;
  bool owns_;
};

class writer_shared_lock {
 public:
  writer_shared_lock(const writer_shared_lock&) = delete;
  writer_shared_lock(writer_shared_lock&&) = delete;

  writer_shared_lock& operator=(const writer_shared_lock&) = delete;
  writer_shared_lock& operator=(writer_shared_lock&&) = delete;

  explicit writer_shared_lock(group_shared_mutex& mutex) : mutex_(&mutex) {
    mutex_->lock_write();
    owns_ = true;
  }

  explicit writer_shared_lock(group_shared_mutex& mutex, std::defer_lock_t)
      : mutex_(&mutex), owns_(false) {}

  ~writer_shared_lock() {
    if (owns_) {
      mutex_->unlock_write();
    }
  }

  void lock() noexcept {
    if (!owns_) {
      mutex_->lock_write();
      owns_ = true;
    }
  }

  bool owns_lock() const noexcept { return owns_; }

 private:
  group_shared_mutex* const mutex_;
  bool owns_;
};

class write_read_lock {
 public:
  write_read_lock(const write_read_lock&) = delete;
  write_read_lock(write_read_lock&&) = delete;

  write_read_lock& operator=(const write_read_lock&) = delete;
  write_read_lock& operator=(write_read_lock&&) = delete;

  explicit write_read_lock(group_shared_mutex& mutex) : mutex_(&mutex) {
    mutex_->lock_write_read();
    owns_ = true;
  }

  explicit write_read_lock(group_shared_mutex& mutex, std::defer_lock_t) noexcept
      : mutex_(&mutex), owns_(false) {}

  ~write_read_lock() {
    if (owns_) {
      mutex_->unlock_write_read();
    }
  }

  void lock() {
    assert(!owns_ && "[write_read_lock] trying to lock twice!");
    mutex_->lock_write_read();
    owns_ = true;
  }

  bool owns_lock() const noexcept { return owns_; }

 private:
  group_shared_mutex* const mutex_;
  bool owns_;
};

}  // namespace merlin
}  // namespace nv