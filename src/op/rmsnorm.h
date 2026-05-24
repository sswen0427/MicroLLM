#pragma once

#include <absl/status/status.h>

#include "layer.h"
namespace op {
class RmsNormLayer : public LayerParam {
 public:
  explicit RmsNormLayer(base::DeviceType device_type, int32_t dim);

  absl::Status check() const override;

  absl::Status forward() override;

 private:
  int32_t dim_ = 0;
};
}  // namespace op
