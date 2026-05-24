#pragma once

#include <absl/status/status.h>

#include "layer.h"
namespace op {
class RoPELayer : public Layer {
 public:
  explicit RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim,
                     int32_t head_size);

  absl::Status check() const override;

  absl::Status forward() override;

 private:
  int32_t dim_ = 0;
  int32_t kv_dim_ = 0;
  int32_t head_size_ = 0;
};
}  // namespace op
