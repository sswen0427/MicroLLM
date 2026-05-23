#include "io/safetensors_reader.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "glog/logging.h"

namespace io {
namespace {

absl::StatusOr<safetensors::tensor_t> FindTensor(
    const safetensors::safetensors_t& safetensors, std::string_view name) {
  safetensors::tensor_t tensor;
  if (!safetensors.tensors.at(std::string(name), &tensor)) {
    return absl::NotFoundError(
        absl::StrCat("Tensor not found in safetensors file: ", name));
  }
  return tensor;
}

}  // namespace

SafetensorsReader::SafetensorsReader(std::string path)
    : path_(std::move(path)) {}

absl::StatusOr<std::unique_ptr<SafetensorsReader>> SafetensorsReader::Open(
    std::string path) {
  auto reader = std::unique_ptr<SafetensorsReader>(
      new SafetensorsReader(std::move(path)));

  std::string warn;
  std::string err;
  if (!safetensors::mmap_from_file(reader->path_, &reader->safetensors_, &warn,
                                   &err)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to open safetensors file: ", reader->path_,
                     err.empty() ? "" : absl::StrCat(", error: ", err)));
  }

  if (!warn.empty()) {
    // Keep warnings observable without making them fatal; malformed files
    // should already be rejected by mmap_from_file or validate_data_offsets.
    LOG(WARNING) << "safetensors warning for " << reader->path_ << ": " << warn;
  }

  std::string offset_error;
  if (!safetensors::validate_data_offsets(reader->safetensors_, offset_error)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid safetensors data offsets in ", reader->path_,
        offset_error.empty() ? "" : absl::StrCat(", error: ", offset_error)));
  }

  return reader;
}

const std::string& SafetensorsReader::path() const { return path_; }

std::size_t SafetensorsReader::tensor_count() const {
  return safetensors_.tensors.size();
}

std::vector<std::string> SafetensorsReader::tensor_names() const {
  return safetensors_.tensors.keys();
}

absl::StatusOr<SafetensorTensorInfo> SafetensorsReader::tensor_info(
    std::string_view name) const {
  auto tensor_or = FindTensor(safetensors_, name);
  if (!tensor_or.ok()) {
    return tensor_or.status();
  }

  const auto& tensor = *tensor_or;
  SafetensorTensorInfo info;
  info.name = std::string(name);
  info.dtype = safetensors::get_dtype_str(tensor.dtype);
  info.shape = tensor.shape;
  info.data_offsets = tensor.data_offsets;
  info.byte_size = tensor.data_offsets[1] - tensor.data_offsets[0];
  return info;
}

absl::StatusOr<const std::uint8_t*> SafetensorsReader::tensor_data(
    std::string_view name) const {
  auto tensor_or = FindTensor(safetensors_, name);
  if (!tensor_or.ok()) {
    return tensor_or.status();
  }

  const auto& tensor = *tensor_or;
  return safetensors_.databuffer_addr + tensor.data_offsets[0];
}

}  // namespace io
