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

#ifndef ABSL_FLAGS_INTERNAL_PARSE_H_
#define ABSL_FLAGS_INTERNAL_PARSE_H_

#include <string>
#include <vector>

#include "absl/flags/declare.h"

ABSL_DECLARE_FLAG(std::vector<std::string>, flagfile);
ABSL_DECLARE_FLAG(std::vector<std::string>, fromenv);
ABSL_DECLARE_FLAG(std::vector<std::string>, tryfromenv);
ABSL_DECLARE_FLAG(std::vector<std::string>, undefok);

namespace absl {
namespace flags_internal {

enum class ArgvListAction { kRemoveParsedArgs, kKeepParsedArgs };
enum class UsageFlagsAction { kHandleUsage, kIgnoreUsage };
enum class OnUndefinedFlag {
  kIgnoreUndefined,
  kReportUndefined,
  kAbortIfUndefined
};

std::vector<char*> ParseCommandLineImpl(int argc, char* argv[],
                                        ArgvListAction arg_list_act,
                                        UsageFlagsAction usage_flag_act,
                                        OnUndefinedFlag on_undef_flag);

}  // namespace flags_internal
}  // namespace absl

#endif  // ABSL_FLAGS_INTERNAL_PARSE_H_
