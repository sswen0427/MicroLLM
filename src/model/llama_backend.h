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
  std::vector<float> key;
  std::vector<float> value;
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
  absl::StatusOr<LlamaForwardResult> ForwardToken(const LlamaHfModel &model,
                                                  LlamaForwardState &state,
                                                  int32_t token_id,
                                                  int32_t position) const;

protected:
  virtual void Embedding(const tensor::Tensor &weight, int32_t token_id,
                         std::vector<float> &output) const = 0;
  virtual tensor::Tensor EmbeddingTensor(const tensor::Tensor &weight,
                                         int32_t token_id) const = 0;
  virtual void RmsNorm(const std::vector<float> &input,
                       const tensor::Tensor &weight, double eps,
                       std::vector<float> &output) const = 0;
  virtual tensor::Tensor RmsNormTensor(const tensor::Tensor &input,
                                       const tensor::Tensor &weight,
                                       double eps) const = 0;
  virtual void MatVec(const tensor::Tensor &weight,
                      const std::vector<float> &input,
                      std::vector<float> &output) const = 0;
  virtual tensor::Tensor MatVecTensor(const tensor::Tensor &weight,
                                      const tensor::Tensor &input) const = 0;
  virtual void ApplyRopeToHeads(std::vector<float> &values, int32_t head_count,
                                int32_t head_size, int32_t position,
                                double rope_theta) const = 0;
  virtual void StoreKvCache(const std::vector<float> &key,
                            const std::vector<float> &value, int32_t position,
                            int32_t max_position, int32_t kv_dim,
                            std::vector<float> &key_cache,
                            std::vector<float> &value_cache) const = 0;
  virtual void AttentionWithCache(const std::vector<float> &query,
                                  const std::vector<float> &key_cache,
                                  const std::vector<float> &value_cache,
                                  int32_t position, int32_t head_count,
                                  int32_t head_size, int32_t kv_dim,
                                  int32_t kv_mul,
                                  std::vector<float> &output) const = 0;
  virtual void SwiGlu(const std::vector<float> &gate,
                      const std::vector<float> &up,
                      std::vector<float> &output) const = 0;
  virtual tensor::Tensor SwiGluTensor(const tensor::Tensor &gate,
                                      const tensor::Tensor &up) const = 0;
  virtual void AddInPlace(std::vector<float> &left,
                          const std::vector<float> &right) const = 0;
  virtual void AddInPlaceTensor(tensor::Tensor &left,
                                const tensor::Tensor &right) const = 0;
  virtual int32_t ArgMaxToken(const tensor::Tensor &logits) const = 0;
};

std::unique_ptr<LlamaBackend> CreateLlamaBackend(base::DeviceType device_type);

} // namespace model
