#include "lookup_table_types.h"

namespace tensorflow {
namespace recommenders_addons {
namespace lookup_table {

std::string TFRA_DataTypeString(TFRA_DataType dtype) {
  switch (dtype) {
    case DT_INVALID:
      return "INVALID";
    case DT_FLOAT:
      return "float";
    case DT_DOUBLE:
      return "double";
    case DT_INT32:
      return "int32";
    case DT_UINT32:
      return "uint32";
    case DT_UINT8:
      return "uint8";
    case DT_UINT16:
      return "uint16";
    case DT_INT16:
      return "int16";
    case DT_INT8:
      return "int8";
    case DT_STRING:
      return "string";
    case DT_COMPLEX64:
      return "complex64";
    case DT_COMPLEX128:
      return "complex128";
    case DT_INT64:
      return "int64";
    case DT_UINT64:
      return "uint64";
    case DT_BOOL:
      return "bool";
    case DT_QINT8:
      return "qint8";
    case DT_QUINT8:
      return "quint8";
    case DT_QUINT16:
      return "quint16";
    case DT_QINT16:
      return "qint16";
    case DT_QINT32:
      return "qint32";
    case DT_BFLOAT16:
      return "bfloat16";
    case DT_HALF:
      return "half";
    case DT_FLOAT8_E5M2:
      return "float8_e5m2";
    case DT_FLOAT8_E4M3FN:
      return "float8_e4m3fn";
    case DT_RESOURCE:
      return "resource";
    case DT_VARIANT:
      return "variant";
    default:
      // LOG(ERROR) << "Unrecognized DataType enum value " << dtype;
      return std::string("unknown dtype enum");
  }
}

void ReadStringFromEnvVar(const std::string& env_var_name,
                          const std::string& default_val, std::string* value) {
  const char* tf_env_var_val = getenv(std::string(env_var_name).c_str());
  if (tf_env_var_val != nullptr) {
    *value = tf_env_var_val;
  } else {
    *value = std::string(default_val);
  }
}

}  // namespace lookup_table
}  // namespace recommenders_addons
}  // namespace tensorflow