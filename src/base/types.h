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
  kDataTypeFp32 = 1,   // 32-bit floating-point
  kDataTypeInt8 = 2,   // 8-bit integer
  kDataTypeInt32 = 3,  //
  kDataTypeBf16 = 4,   // 16-bit bfloat
  kDataTypeFp16 = 5,   // 16-bit floating-point
};

inline std::size_t DataTypeSize(DataType type) {
  switch (type) {
    case DataType::kDataTypeFp32:
      return 4;
    case DataType::kDataTypeInt8:
      return 1;
    case DataType::kDataTypeInt32:
      return 4;
    case DataType::kDataTypeBf16:
    case DataType::kDataTypeFp16:
      return 2;
    default:
      LOG(FATAL) << "Unknown data type";
      return 0;
  }
}

}  // namespace base
