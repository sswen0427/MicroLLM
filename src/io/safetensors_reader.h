#pragma once

#include <absl/status/statusor.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <safetensors.hh>
#include <string>
#include <string_view>
#include <vector>

namespace io {

struct SafetensorTensorInfo {
  std::string name;
  std::string dtype;
  std::vector<std::size_t> shape;
  std::array<std::size_t, 2> data_offsets{};
  std::size_t byte_size = 0;
};

class SafetensorsReader {
 public:
  static absl::StatusOr<std::unique_ptr<SafetensorsReader>> Open(
      std::string path);

  SafetensorsReader(const SafetensorsReader&) = delete;
  SafetensorsReader& operator=(const SafetensorsReader&) = delete;

  const std::string& path() const;
  std::size_t tensor_count() const;
  std::vector<std::string> tensor_names() const;

  absl::StatusOr<SafetensorTensorInfo> tensor_info(std::string_view name) const;
  absl::StatusOr<const std::uint8_t*> tensor_data(std::string_view name) const;

 private:
  explicit SafetensorsReader(std::string path);

  std::string path_;
  safetensors::safetensors_t safetensors_;
};

}  // namespace io
