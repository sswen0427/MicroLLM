#include "model/llama_safetensors_loader.h"

#include <absl/status/status.h>
#include <absl/strings/str_cat.h>
#include <glog/logging.h>

#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "base/types.h"

namespace model {
namespace {

absl::StatusOr<base::DataType> ToDataType(safetensors::dtype dtype) {
  switch (dtype) {
    case safetensors::kINT8:
      return base::DataType::kDataTypeInt8;
    case safetensors::kINT32:
      return base::DataType::kDataTypeInt32;
    case safetensors::kBFLOAT16:
      return base::DataType::kDataTypeBf16;
    case safetensors::kFLOAT32:
      return base::DataType::kDataTypeFp32;
    case safetensors::kFLOAT16:
      return absl::UnimplementedError(
          "FLOAT16 safetensors are not supported by Tensor yet.");
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported safetensors dtype: ",
                       safetensors::get_dtype_str(dtype)));
  }
}

absl::StatusOr<std::vector<int32_t>> ToTensorDims(
    const std::vector<size_t>& shape) {
  if (shape.empty()) {
    return absl::InvalidArgumentError("Scalar safetensors are not supported.");
  }

  std::vector<int32_t> dims;
  dims.reserve(shape.size());
  for (const size_t dim : shape) {
    if (dim == 0) {
      return absl::InvalidArgumentError(
          "Zero-sized safetensors dimensions are not supported.");
    }
    if (dim > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      return absl::InvalidArgumentError(
          absl::StrCat("Safetensors dimension is too large: ", dim));
    }
    dims.push_back(static_cast<int32_t>(dim));
  }
  return dims;
}

const uint8_t* DataBuffer(const safetensors::safetensors_t& safetensors) {
  if (safetensors.mmaped) {
    return safetensors.databuffer_addr;
  }
  return safetensors.storage.data();
}

size_t DataBufferSize(const safetensors::safetensors_t& safetensors) {
  if (safetensors.mmaped) {
    return safetensors.databuffer_size;
  }
  return safetensors.storage.size();
}

}  // namespace

LlamaSafetensorsLoader::LlamaSafetensorsLoader(
    std::unique_ptr<safetensors::safetensors_t> safetensors)
    : safetensors_(std::move(safetensors)) {}

absl::StatusOr<std::unique_ptr<LlamaSafetensorsLoader>>
LlamaSafetensorsLoader::Open(const std::string& safetensors_path) {
  auto safetensors = std::make_unique<safetensors::safetensors_t>();
  std::string warn;
  std::string err;
  if (!safetensors::mmap_from_file(safetensors_path, safetensors.get(), &warn,
                                   &err)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to open safetensors file: ", safetensors_path,
                     err.empty() ? "" : absl::StrCat(", error: ", err)));
  }
  if (!warn.empty()) {
    LOG(WARNING) << "safetensors warning for " << safetensors_path << ": "
                 << warn;
  }

  std::string offset_error;
  if (!safetensors::validate_data_offsets(*safetensors, offset_error)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid safetensors data offsets in ", safetensors_path,
        offset_error.empty() ? "" : absl::StrCat(", error: ", offset_error)));
  }

  return std::unique_ptr<LlamaSafetensorsLoader>(
      new LlamaSafetensorsLoader(std::move(safetensors)));
}

absl::StatusOr<tensor::Tensor> LlamaSafetensorsLoader::LoadTensor(
    const std::string& tensor_name) const {
  safetensors::tensor_t safetensors_tensor;
  if (!safetensors_->tensors.at(tensor_name, &safetensors_tensor)) {
    return absl::NotFoundError(
        absl::StrCat("Tensor not found in safetensors: ", tensor_name));
  }

  auto data_type_or = ToDataType(safetensors_tensor.dtype);
  if (!data_type_or.ok()) {
    return data_type_or.status();
  }
  auto dims_or = ToTensorDims(safetensors_tensor.shape);
  if (!dims_or.ok()) {
    return dims_or.status();
  }

  const size_t begin = safetensors_tensor.data_offsets[0];
  const size_t end = safetensors_tensor.data_offsets[1];
  if (begin > end || end > DataBufferSize(*safetensors_)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid data offsets for tensor ", tensor_name, ": [",
                     begin, ", ", end, "]"));
  }

  tensor::Tensor output = tensor::Tensor::allocate(
      *data_type_or, *dims_or, base::DeviceType::kDeviceCPU);
  if (output.byte_size() != end - begin) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Tensor byte size mismatch for ", tensor_name, ": expected from shape=",
        output.byte_size(), ", safetensors bytes=", end - begin));
  }

  std::memcpy(output.data<uint8_t>(), DataBuffer(*safetensors_) + begin,
              output.byte_size());
  return output;
}

absl::StatusOr<tensor::Tensor> LlamaSafetensorsLoader::LoadGlobalTensor(
    LlamaTensorKind kind) const {
  return LoadTensor(LlamaTensorName(kind));
}

absl::StatusOr<tensor::Tensor> LlamaSafetensorsLoader::LoadLayerTensor(
    int32_t layer, LlamaTensorKind kind) const {
  return LoadTensor(LlamaLayerTensorName(layer, kind));
}

size_t LlamaSafetensorsLoader::TensorCount() const {
  return safetensors_->tensors.size();
}

}  // namespace model
