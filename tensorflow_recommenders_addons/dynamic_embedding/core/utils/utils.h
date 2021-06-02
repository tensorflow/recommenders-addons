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

#ifndef TFRA_UTILS_H_
#define TFRA_UTILS_H_

namespace tensorflow {
namespace recommenders_addons {

#define CONCAT(X, Y, Z) (#X #Y #Z)
#if TF_VERSION_INTEGER >= 2000
#define DECORATE_OP_NAME(N) CONCAT(TFRA, >, N)
#else
#define DECORATE_OP_NAME(N) CONCAT(Tfr, a, N)
#endif

}  // namespace recommenders_addons
}  // namespace tensorflow

#endif  // TFRA_UTILS_H_
