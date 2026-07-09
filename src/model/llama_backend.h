#pragma once

#include <absl/status/statusor.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "base/types.h"
#include "model/llama_hf_model_loader.h"
#include "model/kv_cache.h"
#include "tensor/tensor.h"

namespace model {

struct LlamaForwardResult {
  tensor::Tensor logits;
  int32_t next_token = -1;
};

struct LlamaForwardProfile {
  int64_t forward_calls = 0;
  int64_t prefill_calls = 0;
  int64_t decode_calls = 0;
  int64_t prefill_tokens = 0;
  int64_t decode_tokens = 0;
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

struct LlamaForwardState {
  int32_t head_size = 0;
  int32_t kv_dim = 0;
  int32_t kv_mul = 0;
  KvCache kv_cache;
  LlamaForwardProfile profile;
};

class LlamaBackend {
 public:
  virtual ~LlamaBackend() = default;

  virtual base::DeviceType device_type() const = 0;

  virtual absl::StatusOr<LlamaForwardResult> Forward(
      const LlamaHfModel &model, const std::vector<int32_t> &token_ids,
      int32_t start_position) = 0;

  virtual const LlamaForwardProfile &profile() const = 0;
};

std::unique_ptr<LlamaBackend> CreateLlamaBackend(const HfLlamaConfig &config,
                                                 base::DeviceType device_type);

}  // namespace model
