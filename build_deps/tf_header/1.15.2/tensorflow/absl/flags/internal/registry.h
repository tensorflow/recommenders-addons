//
// Copyright 2019 The Abseil Authors.
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

#ifndef ABSL_FLAGS_INTERNAL_REGISTRY_H_
#define ABSL_FLAGS_INTERNAL_REGISTRY_H_

#include <functional>
#include <map>
#include <string>

#include "absl/base/macros.h"
#include "absl/flags/internal/commandlineflag.h"

// --------------------------------------------------------------------
// Global flags registry API.

namespace absl {
namespace flags_internal {

// CommandLineFlagInfo holds all information for a flag.
struct CommandLineFlagInfo {
  std::string name;           // the name of the flag
  std::string type;           // DO NOT use. Use flag->IsOfType<T>() instead.
  std::string description;    // the "help text" associated with the flag
  std::string current_value;  // the current value, as a std::string
  std::string default_value;  // the default value, as a std::string
  std::string filename;       // 'cleaned' version of filename holding the flag
  bool has_validator_fn;  // true if RegisterFlagValidator called on this flag

  bool is_default;  // true if the flag has the default value and
                    // has not been set explicitly from the cmdline
                    // or via SetCommandLineOption.

  // nullptr for ABSL_FLAG.  A pointer to the flag's current value
  // otherwise.  E.g., for DEFINE_int32(foo, ...), flag_ptr will be
  // &FLAGS_foo.
  const void* flag_ptr;
};

//-----------------------------------------------------------------------------

void FillCommandLineFlagInfo(CommandLineFlag* flag,
                             CommandLineFlagInfo* result);

//-----------------------------------------------------------------------------

CommandLineFlag* FindCommandLineFlag(absl::string_view name);
CommandLineFlag* FindCommandLineV1Flag(const void* flag_ptr);
CommandLineFlag* FindRetiredFlag(absl::string_view name);

// Executes specified visitor for each non-retired flag in the registry.
// Requires the caller hold the registry lock.
void ForEachFlagUnlocked(std::function<void(CommandLineFlag*)> visitor);
// Executes specified visitor for each non-retired flag in the registry. While
// callback are executed, the registry is locked and can't be changed.
void ForEachFlag(std::function<void(CommandLineFlag*)> visitor);

//-----------------------------------------------------------------------------

// Store the list of all flags in *OUTPUT, sorted by file.
void GetAllFlags(std::vector<CommandLineFlagInfo>* OUTPUT);

//-----------------------------------------------------------------------------

bool RegisterCommandLineFlag(CommandLineFlag*, const void* ptr = nullptr);

//-----------------------------------------------------------------------------
// Retired registrations:
//
// Retired flag registrations are treated specially. A 'retired' flag is
// provided only for compatibility with automated invocations that still
// name it.  A 'retired' flag:
//   - is not bound to a C++ FLAGS_ reference.
//   - has a type and a value, but that value is intentionally inaccessible.
//   - does not appear in --help messages.
//   - is fully supported by _all_ flag parsing routines.
//   - consumes args normally, and complains about type mismatches in its
//     argument.
//   - emits a complaint but does not die (e.g. LOG(ERROR)) if it is
//     accessed by name through the flags API for parsing or otherwise.
//
// The registrations for a flag happen in an unspecified order as the
// initializers for the namespace-scope objects of a program are run.
// Any number of weak registrations for a flag can weakly define the flag.
// One non-weak registration will upgrade the flag from weak to non-weak.
// Further weak registrations of a non-weak flag are ignored.
//
// This mechanism is designed to support moving dead flags into a
// 'graveyard' library.  An example migration:
//
//   0: Remove references to this FLAGS_flagname in the C++ codebase.
//   1: Register as 'retired' in old_lib.
//   2: Make old_lib depend on graveyard.
//   3: Add a redundant 'retired' registration to graveyard.
//   4: Remove the old_lib 'retired' registration.
//   5: Eventually delete the graveyard registration entirely.
//
// Returns bool to enable use in namespace-scope initializers.
// For example:
//
//   static const bool dummy = base::RetiredFlag<int32_t>("myflag");
//
// Or to declare several at once:
//
//   static bool dummies[] = {
//       base::RetiredFlag<std::string>("some_string_flag"),
//       base::RetiredFlag<double>("some_double_flag"),
//       base::RetiredFlag<int32_t>("some_int32_flag")
//   };

// Retire flag with name "name" and type indicated by ops.
bool Retire(FlagOpFn ops, FlagMarshallingOpFn marshalling_ops,
            const char* name);

// Registered a retired flag with name 'flag_name' and type 'T'.
template <typename T>
inline bool RetiredFlag(const char* flag_name) {
  return flags_internal::Retire(flags_internal::FlagOps<T>,
                                flags_internal::FlagMarshallingOps<T>,
                                flag_name);
}

// If the flag is retired, returns true and indicates in |*type_is_bool|
// whether the type of the retired flag is a bool.
// Only to be called by code that needs to explicitly ignore retired flags.
bool IsRetiredFlag(absl::string_view name, bool* type_is_bool);

//-----------------------------------------------------------------------------
// Saves the states (value, default value, whether the user has set
// the flag, registered validators, etc) of all flags, and restores
// them when the FlagSaver is destroyed.
//
// This class is thread-safe.  However, its destructor writes to
// exactly the set of flags that have changed value during its
// lifetime, so concurrent _direct_ access to those flags
// (i.e. FLAGS_foo instead of {Get,Set}CommandLineOption()) is unsafe.

class FlagSaver {
 public:
  FlagSaver();
  ~FlagSaver();

  FlagSaver(const FlagSaver&) = delete;
  void operator=(const FlagSaver&) = delete;

  // Prevents saver from restoring the saved state of flags.
  void Ignore();

 private:
  class FlagSaverImpl* impl_;  // we use pimpl here to keep API steady
};

}  // namespace flags_internal
}  // namespace absl

#endif  // ABSL_FLAGS_INTERNAL_REGISTRY_H_
