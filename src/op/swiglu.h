#pragma once

#include <absl/status/status.h>

#include "layer.h"
namespace op {
class SwiGLULayer : public op::Layer {
 public:
  explicit SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim);

  absl::Status check() const override;

  absl::Status forward() override;

 private:
  int32_t hidden_dim_ = 0;
};
}  // namespace op
