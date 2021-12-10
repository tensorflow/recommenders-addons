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

#ifndef BPV2_HMACCUM_CMD_H
#define BPV2_HMACCUM_CMD_H

#include "redismodule.h"

#define OP_SUCCESS 1
#define OP_FAILURE 0

#define BPV2_ERRORMSG_VALUETYPENOTSUPPORTED "Not supported valueType"
#define BPV2_ERRORMSG_INVALIDEXISTSLENGTH "Invalid exists length"

// COMMAND TYPE
#define BPV2_HMACCUM_CMD "HMACCUM"

// MODULE INFO

#define MODULE_NAME "BPV2"
#define MODULE_VERSION 1

enum valueType {
  DT_DOUBLE, /* double */
  DT_FLOAT,  /* float */
  DT_INT32,  /* int32 */
  DT_INT64,  /* int64 */
  DT_INT8,   /* int8  */
  DT_INVALID
};

#endif
