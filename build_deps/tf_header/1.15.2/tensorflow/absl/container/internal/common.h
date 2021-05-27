// Copyright 2018 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ABSL_CONTAINER_INTERNAL_CONTAINER_H_
#define ABSL_CONTAINER_INTERNAL_CONTAINER_H_

#include <cassert>
#include <type_traits>

#include "absl/meta/type_traits.h"
#include "absl/types/optional.h"

namespace absl {
namespace container_internal {

template <class, class = void>
struct IsTransparent : std::false_type {};
template <class T>
struct IsTransparent<T, absl::void_t<typename T::is_transparent>>
    : std::true_type {};

template <bool is_transparent>
struct KeyArg {
  // Transparent. Forward `K`.
  template <typename K, typename key_type>
  using type = K;
};

template <>
struct KeyArg<false> {
  // Not transparent. Always use `key_type`.
  template <typename K, typename key_type>
  using type = key_type;
};

// The node_handle concept from C++17.
// We specialize node_handle for sets and maps. node_handle_base holds the
// common API of both.
template <typename PolicyTraits, typename Alloc>
class node_handle_base {
 protected:
  using slot_type = typename PolicyTraits::slot_type;

 public:
  using allocator_type = Alloc;

  constexpr node_handle_base() {}
  node_handle_base(node_handle_base&& other) noexcept {
    *this = std::move(other);
  }
  ~node_handle_base() { destroy(); }
  node_handle_base& operator=(node_handle_base&& other) noexcept {
    destroy();
    if (!other.empty()) {
      alloc_ = other.alloc_;
      PolicyTraits::transfer(alloc(), slot(), other.slot());
      other.reset();
    }
    return *this;
  }

  bool empty() const noexcept { return !alloc_; }
  explicit operator bool() const noexcept { return !empty(); }
  allocator_type get_allocator() const { return *alloc_; }

 protected:
  friend struct CommonAccess;

  node_handle_base(const allocator_type& a, slot_type* s) : alloc_(a) {
    PolicyTraits::transfer(alloc(), slot(), s);
  }

  void destroy() {
    if (!empty()) {
      PolicyTraits::destroy(alloc(), slot());
      reset();
    }
  }

  void reset() {
    assert(alloc_.has_value());
    alloc_ = absl::nullopt;
  }

  slot_type* slot() const {
    assert(!empty());
    return reinterpret_cast<slot_type*>(std::addressof(slot_space_));
  }
  allocator_type* alloc() { return std::addressof(*alloc_); }

 private:
  absl::optional<allocator_type> alloc_;
  mutable absl::aligned_storage_t<sizeof(slot_type), alignof(slot_type)>
      slot_space_;
};

// For sets.
template <typename Policy, typename PolicyTraits, typename Alloc,
          typename = void>
class node_handle : public node_handle_base<PolicyTraits, Alloc> {
  using Base = typename node_handle::node_handle_base;

 public:
  using value_type = typename PolicyTraits::value_type;

  constexpr node_handle() {}

  value_type& value() const { return PolicyTraits::element(this->slot()); }

 private:
  friend struct CommonAccess;

  node_handle(const Alloc& a, typename Base::slot_type* s) : Base(a, s) {}
};

// For maps.
template <typename Policy, typename PolicyTraits, typename Alloc>
class node_handle<Policy, PolicyTraits, Alloc,
                  absl::void_t<typename Policy::mapped_type>>
    : public node_handle_base<PolicyTraits, Alloc> {
  using Base = typename node_handle::node_handle_base;

 public:
  using key_type = typename Policy::key_type;
  using mapped_type = typename Policy::mapped_type;

  constexpr node_handle() {}

  auto key() const -> decltype(PolicyTraits::key(this->slot())) {
    return PolicyTraits::key(this->slot());
  }

  mapped_type& mapped() const {
    return PolicyTraits::value(&PolicyTraits::element(this->slot()));
  }

 private:
  friend struct CommonAccess;

  node_handle(const Alloc& a, typename Base::slot_type* s) : Base(a, s) {}
};

// Provide access to non-public node-handle functions.
struct CommonAccess {
  template <typename Node>
  static auto GetSlot(const Node& node) -> decltype(node.slot()) {
    return node.slot();
  }

  template <typename Node>
  static void Reset(Node* node) {
    node->reset();
  }

  template <typename T, typename... Args>
  static T Make(Args&&... args) {
    return T(std::forward<Args>(args)...);
  }
};

// Implement the insert_return_type<> concept of C++17.
template <class Iterator, class NodeType>
struct InsertReturnType {
  Iterator position;
  bool inserted;
  NodeType node;
};

}  // namespace container_internal
}  // namespace absl

#endif  // ABSL_CONTAINER_INTERNAL_CONTAINER_H_
