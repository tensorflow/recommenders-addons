/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TFRA_FILEBUFFER_H_
#define TFRA_FILEBUFFER_H_

#include <stdio.h>
#include <stdlib.h>

#include <stdexcept>
#include <string>

#if GOOGLE_CUDA
#include "cuda_runtime.h"
#endif

namespace filebuffer {

enum MODE { READ = 0, WRITE = 1 };

template <typename T>
class FileBuffer {
 public:
  virtual void Put(const T value) {}
  virtual void Flush() {}
  virtual size_t Fill() { return 0; }
  virtual void Clear() {}
  virtual void Close() {}
  virtual size_t size() { return 0; }
  virtual size_t capacity() { return 0; }
};

template <typename T>
class HostFileBuffer : public FileBuffer<T> {
 public:
  HostFileBuffer(const std::string path, size_t capacity, MODE mode)
      : filepath_(path), capacity_(capacity), mode_(mode) {
    offset_ = 0;
    buf_ = (T*)malloc(capacity_ * sizeof(T));
    if (!buf_) {
      throw std::runtime_error("Failed to allocate HostFileBuffer.");
    }
    if (mode_ == MODE::READ) {
      fp_ = fopen(filepath_.c_str(), "rb");
    } else if (mode == MODE::WRITE) {
      fp_ = fopen(filepath_.c_str(), "wb");
    } else {
      throw std::invalid_argument("File mode must be READ or WRITE");
    }
  }

  void Close() override {
    if (buf_) {
      free(buf_);
      buf_ = nullptr;
    }
    if (fp_) {
      fclose(fp_);
      fp_ = nullptr;
    }
  }

  ~HostFileBuffer() { Close(); }

  void Put(const T value) override {
    buf_[offset_++] = value;
    if (offset_ == capacity_) {
      Flush();
    }
  }

  // Must set capacity to be multiples of n.
  void BatchPut(const T* value, size_t n) {
    for (size_t i = 0; i < n; i++) {
      buf_[offset_++] = value[i];
    }
    if (offset_ == capacity_) {
      Flush();
    }
  }

  void Flush() override {
    if (mode_ != MODE::WRITE) {
      throw std::invalid_argument(
          "Can only flush buffer created in WRITE mode.");
    }
    if (offset_ == 0) return;
    size_t nwritten = fwrite(buf_, sizeof(T), offset_, fp_);
    if (nwritten != offset_) {
      throw std::runtime_error("write to " + filepath_ + " expecting " +
                               std::to_string(offset_) + " bytes, but write " +
                               std::to_string(nwritten) + " bytes.");
    }
    offset_ = 0;
  }

  size_t Fill() override {
    offset_ = fread(buf_, sizeof(T), capacity_ - offset_, fp_);
    return offset_;
  }

  void Clear() override { offset_ = 0; }

  T operator[](size_t i) { return buf_[i]; }

  size_t size() override { return offset_; }
  size_t capacity() override { return capacity_; }

  void set_offset(size_t offset) { offset_ = offset; }

 private:
  const std::string filepath_;
  FILE* fp_;
  T* buf_;
  size_t capacity_;
  size_t offset_;
  MODE mode_;
};

#if GOOGLE_CUDA

#ifndef CUDACHECK
#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)
#endif

template <typename T>
class DeviceFileBuffer : public FileBuffer<T> {
 public:
  DeviceFileBuffer(const std::string path, size_t size, MODE mode)
      : filepath_(path), capacity_(size), mode_(mode) {
    offset_ = 0;
    CUDACHECK(cudaMallocHost(&buf_, capacity_ * sizeof(T)));
    if (!buf_) {
      throw std::runtime_error("Failed to allocate DeviceFileBuffer");
    }
    if (mode_ == MODE::READ) {
      fp_ = fopen(filepath_.c_str(), "rb");
    } else if (mode == MODE::WRITE) {
      fp_ = fopen(filepath_.c_str(), "wb");
    } else {
      throw std::invalid_argument("File mode must be READ or WRITE");
    }
  }

  ~DeviceFileBuffer() { Close(); }

  void BatchPut(T* value, size_t n, cudaStream_t stream) {
    CUDACHECK(cudaMemcpyAsync(buf_, value, sizeof(T) * n,
                              cudaMemcpyDeviceToHost, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    offset_ += n;
    Flush();
  }

  void Flush() override {
    if (mode_ != MODE::WRITE) {
      throw std::invalid_argument(
          "Can only flush buffer created in WRITE mode");
    }
    if (offset_ == 0) return;
    size_t nwritten = fwrite(buf_, sizeof(T), offset_, fp_);
    if (nwritten != offset_) {
      throw std::runtime_error("write to " + filepath_ + " expecting " +
                               std::to_string(offset_) + " bytes, but write " +
                               std::to_string(nwritten) + " bytes.");
    }
    offset_ = 0;
  }

  size_t Fill() override {
    offset_ = fread(buf_, sizeof(T), capacity_ - offset_, fp_);
    return offset_;
  }

  void Clear() override { offset_ = 0; }

  void Close() override {
    if (buf_) {
      CUDACHECK(cudaFreeHost(buf_));
      buf_ = nullptr;
    }
    if (fp_) {
      fclose(fp_);
      fp_ = nullptr;
    }
  }

  T* data() { return buf_; }
  size_t size() override { return offset_; }
  size_t capacity() override { return capacity_; }

 private:
  const std::string filepath_;
  FILE* fp_;
  T* buf_;
  size_t capacity_;
  size_t offset_;
  MODE mode_;
};

}  // namespace filebuffer

#endif  // GOOGLE_CUDA

#endif  // TFRA_FILEBUFFER_H_
