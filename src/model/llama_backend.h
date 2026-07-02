#pragma once

#include <absl/status/statusor.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "base/types.h"
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

struct LlamaLayerCache {
  tensor::Tensor key;
  tensor::Tensor value;
};

struct LlamaForwardState {
  int32_t head_size = 0;
  int32_t kv_dim = 0;
  int32_t kv_mul = 0;
  std::vector<LlamaLayerCache> layer_caches;
  LlamaForwardProfile profile;
};

class LlamaBackend {
 public:
  virtual ~LlamaBackend() = default;

  virtual base::DeviceType device_type() const = 0;
  virtual absl::StatusOr<LlamaForwardResult> ForwardToken(
      const LlamaHfModel &model, int32_t token_id, int32_t position) = 0;
  virtual const LlamaForwardProfile &profile() const = 0;
};

std::unique_ptr<LlamaBackend> CreateLlamaBackend(const HfLlamaConfig &config,
                                                 base::DeviceType device_type);

}  // namespace model
