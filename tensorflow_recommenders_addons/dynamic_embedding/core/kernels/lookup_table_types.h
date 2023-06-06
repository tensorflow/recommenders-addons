#ifndef TFRA_CORE_KERNELS_UTILS_LOOKUP_TABLE_TYPES_H_
#define TFRA_CORE_KERNELS_UTILS_LOOKUP_TABLE_TYPES_H_
#include <string>
#include <cstdint>
// #include <string_view>

namespace tensorflow {
namespace recommenders_addons {
namespace lookup_table {

using int8 = std::int8_t;
using uint8 = std::uint8_t;
using int16 = std::int16_t;
using uint16 = std::uint16_t;
using int32 = std::int32_t;
using uint32 = std::uint32_t;

enum StatusCode {
  OK = 0,
  CANCELLED,
  UNKNOWN,
  INVALID_ARGUMENT,
  DEADLINE_EXCEEDED,
  NOT_FOUND,
  ALREADY_EXISTS,
  PERMISSION_DENIED,
  UNAUTHENTICATED,
  RESOURCE_EXHAUSTED,
  FAILED_PRECONDITION,
  ABORTED,
  OUT_OF_RANGE,
  UNIMPLEMENTED,
  INTERNAL,
  UNAVAILABLE,
  DATA_LOSS,
};

class TFRA_Status {
 public:
  TFRA_Status() : code_(StatusCode::OK) {}
  TFRA_Status(StatusCode code, const std::string& message)
      : code_(code), message_(message) {}
  StatusCode code() const { return code_; }
  std::string message() const { return message_; }
  bool ok() const { return StatusCode::OK == code_; }

  static TFRA_Status OK() { return TFRA_Status(); }

 private:
  StatusCode code_;
  std::string message_;
};

#define TFRA_REQUIRES_OK(tfra_status) \
  do {                                \
    TFRA_Status _s = (tfra_status);   \
    if (!_s.ok()) {                   \
      return _s;                      \
    }                                 \
  } while (0)

enum TFRA_DataType {
  // Not a legal value for TFRA_DataType.  Used to indicate a TFRA_DataType field
  // has not been set.
  DT_INVALID = 0,

  // Data types that all computation devices are expected to be
  // capable to support.
  DT_FLOAT = 1,
  DT_DOUBLE = 2,
  DT_INT32 = 3,
  DT_UINT8 = 4,
  DT_INT16 = 5,
  DT_INT8 = 6,
  DT_STRING = 7,
  DT_COMPLEX64 = 8,  // Single-precision complex
  DT_INT64 = 9,
  DT_BOOL = 10,
  DT_QINT8 = 11,     // Quantized int8
  DT_QUINT8 = 12,    // Quantized uint8
  DT_QINT32 = 13,    // Quantized int32
  DT_BFLOAT16 = 14,  // Float32 truncated to 16 bits.
  DT_QINT16 = 15,    // Quantized int16
  DT_QUINT16 = 16,   // Quantized uint16
  DT_UINT16 = 17,
  DT_COMPLEX128 = 18,  // Double-precision complex
  DT_HALF = 19,
  DT_RESOURCE = 20,
  DT_VARIANT = 21,  // Arbitrary C++ data types
  DT_UINT32 = 22,
  DT_UINT64 = 23,
  DT_FLOAT8_E5M2 = 24,    // 5 exponent bits, 3 mantissa bits.
  DT_FLOAT8_E4M3FN = 25,  // 4 exponent bits, 2 mantissa bits, finite-only, with
                          // 2 NaNs (0bS1111111).

  // Do not use!  These are only for parameters.  Every enum above
  // should have a corresponding value below (verified by types_test).
  // DT_FLOAT_REF = 101;
  // DT_DOUBLE_REF = 102;
  // DT_INT32_REF = 103;
  // DT_UINT8_REF = 104;
  // DT_INT16_REF = 105;
  // DT_INT8_REF = 106;
  // DT_STRING_REF = 107;
  // DT_COMPLEX64_REF = 108;
  // DT_INT64_REF = 109;
  // DT_BOOL_REF = 110;
  // DT_QINT8_REF = 111;
  // DT_QUINT8_REF = 112;
  // DT_QINT32_REF = 113;
  // DT_BFLOAT16_REF = 114;
  // DT_QINT16_REF = 115;
  // DT_QUINT16_REF = 116;
  // DT_UINT16_REF = 117;
  // DT_COMPLEX128_REF = 118;
  // DT_HALF_REF = 119;
  // DT_RESOURCE_REF = 120;
  // DT_VARIANT_REF = 121;
  // DT_UINT32_REF = 122;
  // DT_UINT64_REF = 123;
  // DT_FLOAT8_E5M2_REF = 124;
  // DT_FLOAT8_E4M3FN_REF = 125; 
};

std::string TFRA_DataTypeString(TFRA_DataType dtype);

// Validates type T for whether it is a supported TFRA_DataType.
template <class T>
struct IsValidDataType;

// TFRA_DataTypeToEnum<T>::v() and TFRA_DataTypeToEnum<T>::value are the TFRA_DataType
// constants for T, e.g. TFRA_DataTypeToEnum<float>::v() is DT_FLOAT.
template <class T>
struct TFRA_DataTypeToEnum {
  static_assert(IsValidDataType<T>::value, "Specified Data Type not supported");
};  // Specializations below

// TFRA_EnumToDataType<VALUE>::Type is the type for TFRA_DataType constant VALUE, e.g.
// TFRA_EnumToDataType<DT_FLOAT>::Type is float.
template <TFRA_DataType VALUE>
struct TFRA_EnumToDataType {};  // Specializations below

// Template specialization for both TFRA_DataTypeToEnum and TFRA_EnumToDataType.
// static TFRA_DataType ref() { return MakeRefType(ENUM); }

#define MATCH_TYPE_AND_ENUM(TYPE, ENUM)     \
  template <>                               \
  struct TFRA_DataTypeToEnum<TYPE> {             \
    static TFRA_DataType v() { return ENUM; }    \
    static constexpr TFRA_DataType value = ENUM; \
  };                                        \
  template <>                               \
  struct IsValidDataType<TYPE> {            \
    static constexpr bool value = true;     \
  };                                        \
  template <>                               \
  struct TFRA_EnumToDataType<ENUM> {             \
    typedef TYPE Type;                      \
  }

MATCH_TYPE_AND_ENUM(float, DT_FLOAT);
MATCH_TYPE_AND_ENUM(double, DT_DOUBLE);
MATCH_TYPE_AND_ENUM(int32, DT_INT32);
MATCH_TYPE_AND_ENUM(uint32, DT_UINT32);
MATCH_TYPE_AND_ENUM(uint16, DT_UINT16);
MATCH_TYPE_AND_ENUM(uint8, DT_UINT8);
MATCH_TYPE_AND_ENUM(int16, DT_INT16);
MATCH_TYPE_AND_ENUM(int8, DT_INT8);
// MATCH_TYPE_AND_ENUM(std::string_view, DT_STRING);
// MATCH_TYPE_AND_ENUM(complex64, DT_COMPLEX64);
// MATCH_TYPE_AND_ENUM(complex128, DT_COMPLEX128);
MATCH_TYPE_AND_ENUM(bool, DT_BOOL);
// MATCH_TYPE_AND_ENUM(qint8, DT_QINT8);
// MATCH_TYPE_AND_ENUM(quint8, DT_QUINT8);
// MATCH_TYPE_AND_ENUM(qint16, DT_QINT16);
// MATCH_TYPE_AND_ENUM(quint16, DT_QUINT16);
// MATCH_TYPE_AND_ENUM(qint32, DT_QINT32);
// MATCH_TYPE_AND_ENUM(bfloat16, DT_BFLOAT16);
// MATCH_TYPE_AND_ENUM(Eigen::half, DT_HALF);
// MATCH_TYPE_AND_ENUM(float8_e5m2, DT_FLOAT8_E5M2);
// MATCH_TYPE_AND_ENUM(float8_e4m3fn, DT_FLOAT8_E4M3FN);
// MATCH_TYPE_AND_ENUM(ResourceHandle, DT_RESOURCE);
// MATCH_TYPE_AND_ENUM(Variant, DT_VARIANT);

template <>
struct TFRA_DataTypeToEnum<long> {
  static TFRA_DataType v() { return value; }
  // static TFRA_DataType ref() { return MakeRefType(value); }
  static constexpr TFRA_DataType value = sizeof(long) == 4 ? DT_INT32 : DT_INT64;
};
template <>
struct IsValidDataType<long> {
  static constexpr bool value = true;
};
template <>
struct TFRA_EnumToDataType<DT_INT64> {
  typedef int64_t Type;
};

template <>
struct TFRA_DataTypeToEnum<unsigned long> {
  static TFRA_DataType v() { return value; }
  // static TFRA_DataType ref() { return MakeRefType(value); }
  static constexpr TFRA_DataType value =
      sizeof(unsigned long) == 4 ? DT_UINT32 : DT_UINT64;
};
template <>
struct IsValidDataType<unsigned long> {
  static constexpr bool value = true;
};
template <>
struct TFRA_EnumToDataType<DT_UINT64> {
  // typedef tensorflow::uint64 Type;
  typedef uint64_t Type;
};

template <>
struct TFRA_DataTypeToEnum<long long> {
  static TFRA_DataType v() { return DT_INT64; }
  // static TFRA_DataType ref() { return MakeRefType(DT_INT64); }
  static constexpr TFRA_DataType value = DT_INT64;
};
template <>
struct IsValidDataType<long long> {
  static constexpr bool value = true;
};

template <>
struct TFRA_DataTypeToEnum<unsigned long long> {
  static TFRA_DataType v() { return DT_UINT64; }
  // static TFRA_DataType ref() { return MakeRefType(DT_UINT64); }
  static constexpr TFRA_DataType value = DT_UINT64;
};
template <>
struct IsValidDataType<unsigned long long> {
  static constexpr bool value = true;
};

#undef MATCH_TYPE_AND_ENUM

void ReadStringFromEnvVar(const std::string& env_var_name,
                          const std::string& default_val, std::string* value);

}  // namespace lookup_table
}  // namespace recommenders_addons
}  // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_UTILS_LOOKUP_TABLE_TYPES_H_
