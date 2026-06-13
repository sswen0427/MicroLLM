#include "model/llama_backend_util.h"

#include <glog/logging.h>

#include <cstdint>
#include <safetensors.hh>

namespace model {

float TensorElementAsFloat(const tensor::Tensor& tensor, size_t offset) {
  switch (tensor.data_type()) {
    case base::DataType::kDataTypeFp32:
      return tensor.data<float>()[offset];
    case base::DataType::kDataTypeFp16:
      return safetensors::fp16_to_float(tensor.data<uint16_t>()[offset]);
    case base::DataType::kDataTypeBf16:
      return safetensors::bfloat16_to_float(tensor.data<uint16_t>()[offset]);
    default:
      LOG(FATAL) << "Unsupported floating point data type: "
                 << static_cast<int>(tensor.data_type());
  }
  return 0.0f;
}

}  // namespace model
