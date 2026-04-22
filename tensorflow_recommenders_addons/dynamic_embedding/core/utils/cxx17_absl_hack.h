#ifndef TFRA_ABSL_CXX17_HACK_H_
#define TFRA_ABSL_CXX17_HACK_H_

// When compiling with -std=c++17 against TensorFlow 2.9 (which is C++14),
// Abseil headers change the signature of string_view to std::string_view,
// causing ABI linking errors with libtensorflow_framework.so.2 .
// By forcibly undefining these macros after evaluating absl/base/config.h,
// we ensure the C++14 absl::string_view (lts_20211102) is used,
// maintaining binary compatibility while allowing tools (like parlayhash)
// to utilize C++17.
#include "absl/base/config.h"

#undef ABSL_USES_STD_ANY
#undef ABSL_USES_STD_OPTIONAL
#undef ABSL_USES_STD_VARIANT
#undef ABSL_USES_STD_STRING_VIEW

#undef ABSL_HAVE_STD_ANY
#undef ABSL_HAVE_STD_OPTIONAL
#undef ABSL_HAVE_STD_VARIANT
#undef ABSL_HAVE_STD_STRING_VIEW

#endif // TFRA_ABSL_CXX17_HACK_H_
