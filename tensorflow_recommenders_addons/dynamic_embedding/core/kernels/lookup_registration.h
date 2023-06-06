#ifndef TFRA_CORE_KERNELS_LOOKUP_REGISTRATION_H_
#define TFRA_CORE_KERNELS_LOOKUP_REGISTRATION_H_

#include <string.h>

#include <type_traits>
#include <utility>

// Compiler attributes
#if (defined(__GNUC__) || defined(__APPLE__)) && !defined(SWIG)
// Compiler supports GCC-style attributes
#define TFRA_ATTRIBUTE_UNUSED __attribute__((unused))
#elif defined(_MSC_VER)
// Non-GCC equivalents
#define TFRA_ATTRIBUTE_UNUSED
#else
// Non-GCC equivalents
#define TFRA_ATTRIBUTE_UNUSED
#endif

namespace tensorflow {
namespace recommenders_addons {

// An InitOnStartupMarker is 'initialized' on program startup, purely for the
// side-effects of that initialization - the struct itself is empty. (The type
// is expected to be used to define globals.)
//
// The '<<' operator should be used in initializer expressions to specify what
// to run on startup. The following values are accepted:
//   - An InitOnStartupMarker. Example:
//      InitOnStartupMarker F();
//      InitOnStartupMarker const kInitF =
//        InitOnStartupMarker{} << F();
//   - Something to call, which returns an InitOnStartupMarker. Example:
//      InitOnStartupMarker const kInit =
//        InitOnStartupMarker{} << []() { G(); return
//
// See also: TF_INIT_ON_STARTUP_IF
struct InitOnStartupMarker {
  constexpr InitOnStartupMarker operator<<(InitOnStartupMarker) const {
    return *this;
  }

  template <typename T>
  constexpr InitOnStartupMarker operator<<(T&& v) const {
    return std::forward<T>(v)();
  }
};

// Conditional initializer expressions for InitOnStartupMarker:
//   TF_INIT_ON_STARTUP_IF(cond) << f
// If 'cond' is true, 'f' is evaluated (and called, if applicable) on startup.
// Otherwise, 'f' is *not evaluated*. Note that 'cond' is required to be a
// constant-expression, and so this approximates #ifdef.
//
// The implementation uses the ?: operator (!cond prevents evaluation of 'f').
// The relative precedence of ?: and << is significant; this effectively expands
// to (see extra parens):
//   !cond ? InitOnStartupMarker{} : (InitOnStartupMarker{} << f)
//
// Note that although forcing 'cond' to be a constant-expression should not
// affect binary size (i.e. the same optimizations should apply if it 'happens'
// to be one), it was found to be necessary (for a recent version of clang;
// perhaps an optimizer bug).
//
// The parens are necessary to hide the ',' from the preprocessor; it could
// otherwise act as a macro argument separator.
#define TF_INIT_ON_STARTUP_IF(cond)                \
  (::std::integral_constant<bool, !(cond)>::value) \
      ? ::tensorflow::recommenders_addons::InitOnStartupMarker{}        \
      : ::tensorflow::recommenders_addons::InitOnStartupMarker {}

// Wrapper for generating unique IDs (for 'anonymous' InitOnStartup definitions)
// using __COUNTER__. The new ID (__COUNTER__ already expanded) is provided as a
// macro argument.
//
// Usage:
//   #define M_IMPL(id, a, b) ...
//   #define M(a, b) TF_NEW_ID_FOR_INIT(M_IMPL, a, b)
#define TF_NEW_ID_FOR_INIT_2(m, c, ...) m(c, __VA_ARGS__)
#define TF_NEW_ID_FOR_INIT_1(m, c, ...) TF_NEW_ID_FOR_INIT_2(m, c, __VA_ARGS__)
#define TF_NEW_ID_FOR_INIT(m, ...) \
  TF_NEW_ID_FOR_INIT_1(m, __COUNTER__, __VA_ARGS__)

}   // namespace recommenders_addons
}   // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_LOOKUP_REGISTRATION_H_