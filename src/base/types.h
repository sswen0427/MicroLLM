#pragma once

#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include <cstddef>
#include <cstdint>

namespace base {
enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
  kDeviceCUDA = 2,
};

enum class TokenizerType {
  kEncodeUnknown = -1,
  kEncodeSpe = 0,  // https://github.com/google/sentencepiece
  kEncodeBpe = 1,  // https://zhuanlan.zhihu.com/p/424631681
};

struct CudaConfig {
  cudaStream_t stream;

  ~CudaConfig() {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
};

enum class DataType : uint8_t {
  kDataTypeUnknown = 0,
  kDataTypeFp32 = 1,   // 32-bit floating-point
  kDataTypeInt8 = 2,   // 8-bit integer
  kDataTypeInt32 = 3,  //
  kDataTypeBf16 = 4,   // 16-bit bfloat
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
  } else {
    LOG(FATAL) << "Unknown data type";
    return 0;
  }
}

enum class ModelType : uint8_t {
  kModelTypeUnknown = 0,
  kModelTypeLLama2 = 1,
};

}  // namespace base
