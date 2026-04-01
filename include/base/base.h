#ifndef MICROLLM_INCLUDE_BASE_BASE_H
#define MICROLLM_INCLUDE_BASE_BASE_H

#include <cstdint>

enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
};

enum class DataType : uint8_t {
  kDataTypeUnknown = 0,
  kDataTypeFp32 = 1,  // 32-bit floating-point
  kDataTypeInt8 = 2,  // 8-bit integer
};

enum class ModelType : uint8_t {
  kModelTypeUnknown = 0,
  kModelTypeLLama2 = 1,
};

inline std::size_t DataTypeSize(DataType type) {
  if (type == DataType::kDataTypeFp32) {
    return sizeof(float);
  } else if (type == DataType::kDataTypeInt8) {
    return sizeof(int8_t);
  } else {
    return 0;
  }
}

enum class Status : uint8_t {
  kSuccess = 0,
  kFunctionUnImplemented = 1,
  kPathNotValid = 2,
  kParamReadError = 3,
  kWeightReadError = 4,
};

class Noncopyable {
 public:
  Noncopyable() = default;

  ~Noncopyable() = default;

 private:
  Noncopyable(const Noncopyable&) = delete;

  Noncopyable& operator=(const Noncopyable&) = delete;
};

#endif