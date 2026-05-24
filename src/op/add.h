#pragma once

#include <absl/status/status.h>

#include "base/types.h"
#include "layer.h"
namespace op {
class VecAddLayer : public Layer {
 public:
  explicit VecAddLayer(base::DeviceType device_type);

  absl::Status check() const override;

  absl::Status forward() override;
};
}  // namespace op
