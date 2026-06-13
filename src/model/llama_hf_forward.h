#pragma once

#include <cstdint>
#include <memory>

#include "base/types.h"
#include "model/llama_backend.h"
#include "model/llama_hf_model_loader.h"

namespace model {

class LlamaHfRuntime {
public:
  explicit LlamaHfRuntime(
      const LlamaHfModel &model,
      base::DeviceType device_type = base::DeviceType::kDeviceCPU);

  absl::StatusOr<LlamaForwardResult> ForwardToken(int32_t token_id,
                                                  int32_t position);

  const LlamaForwardProfile &profile() const { return state_.profile; }

private:
  const LlamaHfModel &model_;
  LlamaForwardState state_;
  std::unique_ptr<LlamaBackend> backend_;
};

} // namespace model
