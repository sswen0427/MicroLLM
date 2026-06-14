#include "model/llama_backend.h"

#include <absl/status/status.h>
#include <absl/strings/str_cat.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "base/profile.h"
#include "model/llama_backend_util.h"

namespace model {
namespace {

tensor::Tensor CopyVectorToCpuTensor(const std::vector<float> &values) {
  tensor::Tensor tensor = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {static_cast<int32_t>(values.size())},
      base::DeviceType::kDeviceCPU);
  std::copy(values.begin(), values.end(), tensor.data<float>());
  return tensor;
}

void Embedding(const tensor::Tensor &weight, int32_t token_id,
               std::vector<float> &output);
void RmsNorm(const std::vector<float> &input, const tensor::Tensor &weight,
             double eps, std::vector<float> &output);
void MatVec(const tensor::Tensor &weight, const std::vector<float> &input,
            std::vector<float> &output);
void ApplyRopeToHeads(std::vector<float> &values, int32_t head_count,
                      int32_t head_size, int32_t position, double rope_theta);
void StoreKvCache(const std::vector<float> &key,
                  const std::vector<float> &value, int32_t position,
                  int32_t max_position, int32_t kv_dim,
                  std::vector<float> &key_cache,
                  std::vector<float> &value_cache);
void AttentionWithCache(const std::vector<float> &query,
                        const std::vector<float> &key_cache,
                        const std::vector<float> &value_cache, int32_t position,
                        int32_t head_count, int32_t head_size, int32_t kv_dim,
                        int32_t kv_mul, std::vector<float> &output);
void SwiGlu(const std::vector<float> &gate, const std::vector<float> &up,
            std::vector<float> &output);
void AddInPlace(std::vector<float> &left, const std::vector<float> &right);
int32_t ArgMaxToken(const tensor::Tensor &logits);
void SoftmaxInPlace(std::vector<float> &values);

} // namespace

class CpuLlamaBackend final : public LlamaBackend {
public:
  base::DeviceType device_type() const override;
  absl::StatusOr<LlamaForwardResult>
  ForwardToken(const LlamaHfModel &model, LlamaForwardState &state,
               int32_t token_id, int32_t position) const override;
};

std::unique_ptr<LlamaBackend> CreateCpuLlamaBackend() {
  return std::make_unique<CpuLlamaBackend>();
}

base::DeviceType CpuLlamaBackend::device_type() const {
  return base::DeviceType::kDeviceCPU;
}

absl::StatusOr<LlamaForwardResult>
CpuLlamaBackend::ForwardToken(const LlamaHfModel &model,
                              LlamaForwardState &state, int32_t token_id,
                              int32_t position) const {
  const HfLlamaConfig &config = model.config;
  if (token_id < 0 || token_id >= config.vocab_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("token_id is out of range: ", token_id,
                     ", vocab_size=", config.vocab_size));
  }
  if (position < 0 || position >= config.max_position_embeddings) {
    return absl::InvalidArgumentError(absl::StrCat(
        "position is out of range: ", position,
        ", max_position_embeddings=", config.max_position_embeddings));
  }

  LOG(INFO) << "start LLaMA CPU one-token forward: token_id=" << token_id
            << ", position=" << position;
  state.profile.forward_calls += 1;

  std::vector<float> hidden_state;
  {
    base::ScopedProfile profile(state.profile.embedding_ms);
    Embedding(model.weights.token_embedding, token_id, hidden_state);
  }

  std::vector<float> norm;
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> attention_output;
  std::vector<float> projected_attention;
  std::vector<float> gate;
  std::vector<float> up;
  std::vector<float> activated;
  std::vector<float> projected_ffn;

  for (int32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    const LlamaHfLayerWeights &weights = model.weights.layers[layer];

    {
      base::ScopedProfile profile(state.profile.attention_norm_ms);
      RmsNorm(hidden_state, weights.input_layernorm, config.rms_norm_eps, norm);
    }
    {
      base::ScopedProfile profile(state.profile.qkv_proj_ms);
      MatVec(weights.q_proj, norm, query);
      MatVec(weights.k_proj, norm, key);
      MatVec(weights.v_proj, norm, value);
    }
    {
      base::ScopedProfile profile(state.profile.rope_ms);
      ApplyRopeToHeads(query, config.num_attention_heads, state.head_size,
                       position, config.rope_theta);
      ApplyRopeToHeads(key, config.num_key_value_heads, state.head_size,
                       position, config.rope_theta);
    }
    {
      base::ScopedProfile profile(state.profile.kv_cache_ms);
      StoreKvCache(key, value, position, config.max_position_embeddings,
                   state.kv_dim, state.layer_caches[layer].key,
                   state.layer_caches[layer].value);
    }
    {
      base::ScopedProfile profile(state.profile.attention_ms);
      AttentionWithCache(query, state.layer_caches[layer].key,
                         state.layer_caches[layer].value, position,
                         config.num_attention_heads, state.head_size,
                         state.kv_dim, state.kv_mul, attention_output);
    }
    {
      base::ScopedProfile profile(state.profile.attention_output_proj_ms);
      MatVec(weights.o_proj, attention_output, projected_attention);
    }
    {
      base::ScopedProfile profile(state.profile.attention_residual_ms);
      AddInPlace(hidden_state, projected_attention);
    }
    {
      base::ScopedProfile profile(state.profile.ffn_norm_ms);
      RmsNorm(hidden_state, weights.post_attention_layernorm,
              config.rms_norm_eps, norm);
    }
    {
      base::ScopedProfile profile(state.profile.ffn_up_gate_proj_ms);
      MatVec(weights.gate_proj, norm, gate);
      MatVec(weights.up_proj, norm, up);
    }
    {
      base::ScopedProfile profile(state.profile.swiglu_ms);
      SwiGlu(gate, up, activated);
    }
    {
      base::ScopedProfile profile(state.profile.ffn_down_proj_ms);
      MatVec(weights.down_proj, activated, projected_ffn);
    }
    {
      base::ScopedProfile profile(state.profile.ffn_residual_ms);
      AddInPlace(hidden_state, projected_ffn);
    }
  }

  {
    base::ScopedProfile profile(state.profile.final_norm_ms);
    RmsNorm(hidden_state, model.weights.final_norm, config.rms_norm_eps, norm);
  }

  LlamaForwardResult result;
  std::vector<float> logits;
  {
    base::ScopedProfile profile(state.profile.lm_head_ms);
    MatVec(model.weights.lm_head, norm, logits);
  }
  result.logits = CopyVectorToCpuTensor(logits);
  CHECK_EQ(static_cast<int32_t>(result.logits.size()), config.vocab_size);
  {
    base::ScopedProfile profile(state.profile.argmax_ms);
    result.next_token = ArgMaxToken(result.logits);
  }

  LOG(INFO) << "finish LLaMA CPU one-token forward: next_token="
            << result.next_token;
  return result;
}

namespace {

void Embedding(const tensor::Tensor &weight, int32_t token_id,
               std::vector<float> &output) {
  const int32_t cols = weight.get_dim(1);
  output.resize(cols);
  const size_t row_offset = static_cast<size_t>(token_id) * cols;
  for (int32_t col = 0; col < cols; ++col) {
    output[col] = TensorElementAsFloat(weight, row_offset + col);
  }
}

void RmsNorm(const std::vector<float> &input, const tensor::Tensor &weight,
             double eps, std::vector<float> &output) {
  float square_sum = 0.0f;
  for (const float value : input) {
    square_sum += value * value;
  }
  const float mean_square = square_sum / static_cast<float>(input.size());
  const float scale = 1.0f / std::sqrt(mean_square + static_cast<float>(eps));

  output.resize(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = input[i] * scale * TensorElementAsFloat(weight, i);
  }
}

void MatVec(const tensor::Tensor &weight, const std::vector<float> &input,
            std::vector<float> &output) {
  const int32_t rows = weight.get_dim(0);
  const int32_t cols = weight.get_dim(1);
  CHECK_EQ(static_cast<int32_t>(input.size()), cols);
  output.assign(rows, 0.0f);
  for (int32_t row = 0; row < rows; ++row) {
    float sum = 0.0f;
    const size_t row_offset = static_cast<size_t>(row) * cols;
    for (int32_t col = 0; col < cols; ++col) {
      sum += TensorElementAsFloat(weight, row_offset + col) * input[col];
    }
    output[row] = sum;
  }
}

void ApplyRopeToHeads(std::vector<float> &values, int32_t head_count,
                      int32_t head_size, int32_t position, double rope_theta) {
  CHECK_EQ(static_cast<int32_t>(values.size()), head_count * head_size);
  CHECK_EQ(head_size % 2, 0);
  const int32_t half_head_size = head_size / 2;
  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t head_offset = head * head_size;
    for (int32_t i = 0; i < half_head_size; ++i) {
      const float freq = 1.0f / std::pow(static_cast<float>(rope_theta),
                                         static_cast<float>(2 * i) /
                                             static_cast<float>(head_size));
      const float angle = static_cast<float>(position) * freq;
      const float cos_value = std::cos(angle);
      const float sin_value = std::sin(angle);

      const int32_t first = head_offset + i;
      const int32_t second = head_offset + i + half_head_size;
      const float x0 = values[first];
      const float x1 = values[second];
      values[first] = x0 * cos_value - x1 * sin_value;
      values[second] = x0 * sin_value + x1 * cos_value;
    }
  }
}

void StoreKvCache(const std::vector<float> &key,
                  const std::vector<float> &value, int32_t position,
                  int32_t max_position, int32_t kv_dim,
                  std::vector<float> &key_cache,
                  std::vector<float> &value_cache) {
  CHECK_EQ(static_cast<int32_t>(key.size()), kv_dim);
  CHECK_EQ(static_cast<int32_t>(value.size()), kv_dim);
  CHECK_GE(position, 0);
  CHECK_LT(position, max_position);
  const size_t offset = static_cast<size_t>(position) * kv_dim;
  std::copy(key.begin(), key.end(), key_cache.begin() + offset);
  std::copy(value.begin(), value.end(), value_cache.begin() + offset);
}

void AttentionWithCache(const std::vector<float> &query,
                        const std::vector<float> &key_cache,
                        const std::vector<float> &value_cache, int32_t position,
                        int32_t head_count, int32_t head_size, int32_t kv_dim,
                        int32_t kv_mul, std::vector<float> &output) {
  CHECK_GE(position, 0);
  CHECK_EQ(static_cast<int32_t>(query.size()), head_count * head_size);
  output.assign(static_cast<size_t>(head_count) * head_size, 0.0f);

  std::vector<float> scores(static_cast<size_t>(position) + 1);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t kv_head = head / kv_mul;
    const int32_t query_offset = head * head_size;
    const int32_t output_offset = head * head_size;

    for (int32_t token = 0; token <= position; ++token) {
      const int32_t cache_offset = token * kv_dim + kv_head * head_size;
      float score = 0.0f;
      for (int32_t i = 0; i < head_size; ++i) {
        score += query[query_offset + i] * key_cache[cache_offset + i];
      }
      scores[token] = score * scale;
    }
    SoftmaxInPlace(scores);

    for (int32_t token = 0; token <= position; ++token) {
      const int32_t cache_offset = token * kv_dim + kv_head * head_size;
      const float score = scores[token];
      for (int32_t i = 0; i < head_size; ++i) {
        output[output_offset + i] += score * value_cache[cache_offset + i];
      }
    }
  }
}

void SwiGlu(const std::vector<float> &gate, const std::vector<float> &up,
            std::vector<float> &output) {
  CHECK_EQ(gate.size(), up.size());
  output.resize(gate.size());
  for (size_t i = 0; i < gate.size(); ++i) {
    const float silu = gate[i] / (1.0f + std::exp(-gate[i]));
    output[i] = silu * up[i];
  }
}

void AddInPlace(std::vector<float> &left, const std::vector<float> &right) {
  CHECK_EQ(left.size(), right.size());
  for (size_t i = 0; i < left.size(); ++i) {
    left[i] += right[i];
  }
}

int32_t ArgMaxToken(const tensor::Tensor &logits) {
  CHECK(logits.data_type() == base::DataType::kDataTypeFp32);
  const float *data = logits.data<float>();
  int32_t best = 0;
  float best_value = data[0];
  for (int32_t i = 1; i < static_cast<int32_t>(logits.size()); ++i) {
    if (data[i] > best_value) {
      best = i;
      best_value = data[i];
    }
  }
  return best;
}

void SoftmaxInPlace(std::vector<float> &values) {
  CHECK(!values.empty());
  const float max_value = *std::max_element(values.begin(), values.end());
  float sum = 0.0f;
  for (float &value : values) {
    value = std::exp(value - max_value);
    sum += value;
  }
  for (float &value : values) {
    value /= sum;
  }
}

} // namespace

} // namespace model
