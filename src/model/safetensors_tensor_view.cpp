#include "model/safetensors_tensor_view.h"

#include <absl/status/status.h>
#include <absl/strings/str_cat.h>

#include <cstdint>
#include <limits>
#include <safetensors.hh>
#include <string>
#include <vector>

#include "base/base.h"

namespace model {
namespace {

absl::StatusOr<base::DataType> ToRuntimeDataType(safetensors::dtype dtype) {
  switch (dtype) {
    case safetensors::dtype::kBFLOAT16:
      return base::DataType::kDataTypeBf16;
    case safetensors::dtype::kFLOAT32:
      return base::DataType::kDataTypeFp32;
    case safetensors::dtype::kINT8:
      return base::DataType::kDataTypeInt8;
    case safetensors::dtype::kINT32:
      return base::DataType::kDataTypeInt32;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported safetensors dtype: ",
                       safetensors::get_dtype_str(dtype)));
  }
}

absl::StatusOr<std::vector<int64_t>> ToRuntimeShape(
    const std::vector<size_t>& shape) {
  std::vector<int64_t> runtime_shape;
  runtime_shape.reserve(shape.size());
  for (const size_t dim : shape) {
    if (dim > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      return absl::InvalidArgumentError(
          absl::StrCat("Tensor dimension is too large: ", dim));
    }
    runtime_shape.push_back(static_cast<int64_t>(dim));
  }
  return runtime_shape;
}

const uint8_t* GetDataBuffer(const safetensors::safetensors_t& safetensors) {
  if (safetensors.mmaped) {
    return safetensors.databuffer_addr;
  }
  if (!safetensors.storage.empty()) {
    return safetensors.storage.data();
  }
  return nullptr;
}

}  // namespace

absl::StatusOr<tensor::TensorView> GetSafetensorsTensorView(
    const safetensors::safetensors_t& safetensors,
    const std::string& tensor_name) {
  safetensors::tensor_t safetensors_tensor;
  if (!safetensors.tensors.at(tensor_name, &safetensors_tensor)) {
    return absl::NotFoundError(
        absl::StrCat("Tensor not found in safetensors: ", tensor_name));
  }

  auto data_type_or = ToRuntimeDataType(safetensors_tensor.dtype);
  if (!data_type_or.ok()) {
    return data_type_or.status();
  }
  auto shape_or = ToRuntimeShape(safetensors_tensor.shape);
  if (!shape_or.ok()) {
    return shape_or.status();
  }

  const uint8_t* data_buffer = GetDataBuffer(safetensors);
  if (data_buffer == nullptr) {
    return absl::InvalidArgumentError(
        "Safetensors data buffer is not available.");
  }

  const size_t begin = safetensors_tensor.data_offsets[0];
  const size_t end = safetensors_tensor.data_offsets[1];
  if (begin > end) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid data offsets for tensor ", tensor_name, ": [",
                     begin, ", ", end, "]"));
  }

  tensor::TensorView view{
      .data_type = *data_type_or,
      .device_type = base::DeviceType::kDeviceCPU,
      .shape = *shape_or,
      .data = data_buffer + begin,
      .byte_size = end - begin,
  };
  if (!view.is_valid()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid tensor view metadata for tensor: ", tensor_name));
  }
  return view;
}

}  // namespace model
