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

/* The decorate macro is only used for some old OPs,
we had to compatible with TF1.x naming rule which doesn't support the
"XYZ>AbcEfg", but we can't change the old OPs name directly because some users
already used TFRA release 1 in product environment, and changing name to
"TfraXyz" may cause compatible problem. For new OPs, we should not use
"TFRA>Xyz" at first. */
#define CONCAT_TRIPLE_STRING(X, Y, Z) (#X #Y #Z)
#if TF_VERSION_INTEGER >= 2000
#define PREFIX_OP_NAME(N) CONCAT_TRIPLE_STRING(TFRA, >, N)
#else
#define PREFIX_OP_NAME(N) CONCAT_TRIPLE_STRING(Tfr, a, N)
#endif

/* After TensorFlow version 2.10.0, "Status::OK()" upgraded to "OkStatus()".
This code is for compatibility.*/
#if TF_VERSION_INTEGER >= 2100
#define TFOkStatus OkStatus()
#else
#define TFOkStatus Status::OK()
#endif
}  // namespace recommenders_addons
}  // namespace tensorflow

#endif  // TFRA_UTILS_H_
