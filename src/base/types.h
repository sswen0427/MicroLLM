#pragma once

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>

namespace base {
enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
  kDeviceCUDA = 2,
};

enum class DataType : uint8_t {
  kDataTypeUnknown = 0,
  kDataTypeFp32 = 1,  // 32-bit floating-point
  kDataTypeInt8 = 2,  // 8-bit integer
  kDataTypeInt32 = 3, //
  kDataTypeBf16 = 4,  // 16-bit bfloat
  kDataTypeFp16 = 5,  // 16-bit floating-point
};

inline std::size_t DataTypeSize(DataType type) {
  if (type == DataType::kDataTypeFp32) {
    return 4;
  } else if (type == DataType::kDataTypeInt8) {
    return 1;
  } else if (type == DataType::kDataTypeInt32) {
    return 4;
  } else if (type == DataType::kDataTypeBf16) {
    return 2;
  } else if (type == DataType::kDataTypeFp16) {
    return 2;
  } else {
    LOG(FATAL) << "Unknown data type";
    return 0;
  }
}

} // namespace base
