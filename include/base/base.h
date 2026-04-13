#ifndef MICROLLM_INCLUDE_BASE_BASE_H
#define MICROLLM_INCLUDE_BASE_BASE_H

#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include <string>

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

enum StatusCode : uint8_t {
  kSuccess = 0,
  kFunctionUnImplement = 1,
  kPathNotValid = 2,
  kModelParseError = 3,
  kInternalError = 5,
  kKeyValueHasExist = 6,
  kInvalidArgument = 7,
};

struct CudaConfig {
  cudaStream_t stream;

  ~CudaConfig() {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
};

class Status {
 public:
  Status(int code = StatusCode::kSuccess, std::string err_message = "");

  Status(const Status& other) = default;

  Status& operator=(const Status& other) = default;

  Status& operator=(int code);

  bool operator==(int code) const;

  bool operator!=(int code) const;

  operator int() const;

  operator bool() const;

  int32_t get_err_code() const;

  const std::string& get_err_msg() const;

  void set_err_msg(const std::string& err_msg);

 private:
  int code_ = StatusCode::kSuccess;
  std::string message_;
};

enum class DataType : uint8_t {
  kDataTypeUnknown = 0,
  kDataTypeFp32 = 1,   // 32-bit floating-point
  kDataTypeInt8 = 2,   // 8-bit integer
  kDataTypeInt32 = 3,  //
};

inline std::size_t DataTypeSize(DataType type) {
  if (type == DataType::kDataTypeFp32) {
    return 4;
  } else if (type == DataType::kDataTypeInt8) {
    return 1;
  } else if (type == DataType::kDataTypeInt32) {
    return 4;
  } else {
    LOG(FATAL) << "Unknown data type";
  }
}
}  // namespace base

#endif