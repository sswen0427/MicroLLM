#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "base/base.h"

namespace tensor {

struct TensorView {
  base::DataType data_type = base::DataType::kDataTypeUnknown;
  base::DeviceType device_type = base::DeviceType::kDeviceCPU;
  std::vector<int64_t> shape;
  const void* data = nullptr;
  std::size_t byte_size = 0;

  [[nodiscard]] bool is_valid() const {
    return data_type != base::DataType::kDataTypeUnknown &&
           device_type != base::DeviceType::kDeviceUnknown && data != nullptr &&
           byte_size > 0 && byte_size == expected_byte_size();
  }

  [[nodiscard]] std::size_t expected_byte_size() const {
    if (data_type == base::DataType::kDataTypeUnknown) {
      return 0;
    }
    return element_count() * base::DataTypeSize(data_type);
  }

  [[nodiscard]] std::size_t element_count() const {
    if (shape.empty()) {
      return 0;
    }
    for (const int64_t dim : shape) {
      if (dim <= 0) {
        return 0;
      }
    }
    return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                           [](std::size_t total, int64_t dim) {
                             return total * static_cast<std::size_t>(dim);
                           });
  }
};

}  // namespace tensor
