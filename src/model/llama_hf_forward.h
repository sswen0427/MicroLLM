#pragma once

#include <absl/status/statusor.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "base/types.h"
#include "model/llama_backend.h"
#include "model/llama_hf_model_loader.h"
#include "tensor/tensor.h"

namespace model {

struct LlamaForwardResult {
  tensor::Tensor logits;
  int32_t next_token = -1;
};

struct LlamaForwardProfile {
  int64_t forward_calls = 0;
  double embedding_ms = 0.0;
  double attention_norm_ms = 0.0;
  double qkv_proj_ms = 0.0;
  double rope_ms = 0.0;
  double kv_cache_ms = 0.0;
  double attention_ms = 0.0;
  double attention_output_proj_ms = 0.0;
  double attention_residual_ms = 0.0;
  double ffn_norm_ms = 0.0;
  double ffn_up_gate_proj_ms = 0.0;
  double swiglu_ms = 0.0;
  double ffn_down_proj_ms = 0.0;
  double ffn_residual_ms = 0.0;
  double final_norm_ms = 0.0;
  double lm_head_ms = 0.0;
  double argmax_ms = 0.0;

  void Log() const;
};

class LlamaHfRuntime {
 public:
  explicit LlamaHfRuntime(
      const LlamaHfModel& model,
      base::DeviceType device_type = base::DeviceType::kDeviceCPU);

  absl::StatusOr<LlamaForwardResult> ForwardToken(int32_t token_id,
                                                  int32_t position);

  const LlamaForwardProfile& profile() const { return profile_; }

 private:
  struct LayerCache {
    std::vector<float> key;
    std::vector<float> value;
  };

  const LlamaHfModel& model_;
  int32_t head_size_ = 0;
  int32_t kv_dim_ = 0;
  int32_t kv_mul_ = 0;
  std::vector<LayerCache> layer_caches_;
  LlamaForwardProfile profile_;
  std::unique_ptr<LlamaBackend> backend_;
};

}  // namespace model
