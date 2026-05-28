#include "model/llama_hf_forward.h"

#include <absl/status/status.h>
#include <absl/strings/str_cat.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <safetensors.hh>
#include <vector>

#include "base/profile.h"
#include "base/types.h"

namespace model {
namespace {

float TensorElementAsFloat(const tensor::Tensor& tensor, size_t offset) {
  switch (tensor.data_type()) {
    case base::DataType::kDataTypeFp32:
      return tensor.data<float>()[offset];
    case base::DataType::kDataTypeFp16:
      return safetensors::fp16_to_float(tensor.data<uint16_t>()[offset]);
    case base::DataType::kDataTypeBf16:
      return safetensors::bfloat16_to_float(tensor.data<uint16_t>()[offset]);
    default:
      LOG(FATAL) << "Unsupported floating point data type: "
                 << static_cast<int>(tensor.data_type());
  }
  return 0.0f;
}

void CopyMatrixRow(const tensor::Tensor& matrix, int32_t row,
                   std::vector<float>& output) {
  const int32_t cols = matrix.get_dim(1);
  output.resize(cols);
  const size_t row_offset = static_cast<size_t>(row) * cols;
  for (int32_t col = 0; col < cols; ++col) {
    output[col] = TensorElementAsFloat(matrix, row_offset + col);
  }
}

void MatVec(const tensor::Tensor& weight, const std::vector<float>& input,
            std::vector<float>& output) {
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

void RmsNorm(const std::vector<float>& input, const tensor::Tensor& weight,
             double eps, std::vector<float>& output) {
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

void AddInPlace(std::vector<float>& left, const std::vector<float>& right) {
  CHECK_EQ(left.size(), right.size());
  for (size_t i = 0; i < left.size(); ++i) {
    left[i] += right[i];
  }
}

void SwiGlu(const std::vector<float>& gate, const std::vector<float>& up,
            std::vector<float>& output) {
  CHECK_EQ(gate.size(), up.size());
  output.resize(gate.size());
  for (size_t i = 0; i < gate.size(); ++i) {
    const float silu = gate[i] / (1.0f + std::exp(-gate[i]));
    output[i] = silu * up[i];
  }
}

void ApplyRopeToHeads(std::vector<float>& values, int32_t head_count,
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

void StoreKvCache(const std::vector<float>& key,
                  const std::vector<float>& value, int32_t position,
                  int32_t max_position, int32_t kv_dim,
                  std::vector<float>& key_cache,
                  std::vector<float>& value_cache) {
  CHECK_EQ(static_cast<int32_t>(key.size()), kv_dim);
  CHECK_EQ(static_cast<int32_t>(value.size()), kv_dim);
  CHECK_GE(position, 0);
  CHECK_LT(position, max_position);
  const size_t offset = static_cast<size_t>(position) * kv_dim;
  std::copy(key.begin(), key.end(), key_cache.begin() + offset);
  std::copy(value.begin(), value.end(), value_cache.begin() + offset);
}

void SoftmaxInPlace(std::vector<float>& values) {
  CHECK(!values.empty());
  const float max_value = *std::max_element(values.begin(), values.end());
  float sum = 0.0f;
  for (float& value : values) {
    value = std::exp(value - max_value);
    sum += value;
  }
  for (float& value : values) {
    value /= sum;
  }
}

void AttentionWithCache(const std::vector<float>& query,
                        const std::vector<float>& key_cache,
                        const std::vector<float>& value_cache, int32_t position,
                        int32_t head_count, int32_t head_size, int32_t kv_dim,
                        int32_t kv_mul, std::vector<float>& output) {
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

int32_t ArgMaxToken(const tensor::Tensor& logits) {
  CHECK(logits.data_type() == base::DataType::kDataTypeFp32);
  const float* data = logits.data<float>();
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
}  // namespace

LlamaHfRuntime::LlamaHfRuntime(const LlamaHfModel& model) : model_(model) {
  const HfLlamaConfig& config = model_.config;
  if (config.num_attention_heads > 0) {
    head_size_ = config.hidden_size / config.num_attention_heads;
  }
  kv_dim_ = config.num_key_value_heads * head_size_;
  if (config.num_key_value_heads > 0) {
    kv_mul_ = config.num_attention_heads / config.num_key_value_heads;
  }

  if (config.num_hidden_layers <= 0 || config.max_position_embeddings <= 0 ||
      kv_dim_ <= 0) {
    return;
  }

  layer_caches_.resize(config.num_hidden_layers);
  const size_t cache_size =
      static_cast<size_t>(config.max_position_embeddings) * kv_dim_;
  for (LayerCache& cache : layer_caches_) {
    cache.key.assign(cache_size, 0.0f);
    cache.value.assign(cache_size, 0.0f);
  }
}

absl::StatusOr<LlamaForwardResult> LlamaHfRuntime::ForwardToken(
    int32_t token_id, int32_t position) {
  const HfLlamaConfig& config = model_.config;
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

  LOG(INFO) << "start LLaMA HF one-token forward: token_id=" << token_id
            << ", position=" << position;
  profile_.forward_calls += 1;

  std::vector<float> hidden_state;
  {
    base::ScopedProfile profile(profile_.embedding_ms);
    CopyMatrixRow(model_.weights.token_embedding, token_id, hidden_state);
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
    const LlamaHfLayerWeights& weights = model_.weights.layers[layer];

    {
      base::ScopedProfile profile(profile_.attention_norm_ms);
      RmsNorm(hidden_state, weights.input_layernorm, config.rms_norm_eps, norm);
    }
    {
      base::ScopedProfile profile(profile_.qkv_proj_ms);
      MatVec(weights.q_proj, norm, query);
      MatVec(weights.k_proj, norm, key);
      MatVec(weights.v_proj, norm, value);
    }

    {
      base::ScopedProfile profile(profile_.rope_ms);
      ApplyRopeToHeads(query, config.num_attention_heads, head_size_, position,
                       config.rope_theta);
      ApplyRopeToHeads(key, config.num_key_value_heads, head_size_, position,
                       config.rope_theta);
    }

    {
      base::ScopedProfile profile(profile_.kv_cache_ms);
      StoreKvCache(key, value, position, config.max_position_embeddings,
                   kv_dim_, layer_caches_[layer].key,
                   layer_caches_[layer].value);
    }
    {
      base::ScopedProfile profile(profile_.attention_ms);
      AttentionWithCache(query, layer_caches_[layer].key,
                         layer_caches_[layer].value, position,
                         config.num_attention_heads, head_size_, kv_dim_,
                         kv_mul_, attention_output);
    }
    {
      base::ScopedProfile profile(profile_.attention_output_proj_ms);
      MatVec(weights.o_proj, attention_output, projected_attention);
    }
    {
      base::ScopedProfile profile(profile_.attention_residual_ms);
      AddInPlace(hidden_state, projected_attention);
    }

    {
      base::ScopedProfile profile(profile_.ffn_norm_ms);
      RmsNorm(hidden_state, weights.post_attention_layernorm,
              config.rms_norm_eps, norm);
    }
    {
      base::ScopedProfile profile(profile_.ffn_up_gate_proj_ms);
      MatVec(weights.gate_proj, norm, gate);
      MatVec(weights.up_proj, norm, up);
    }
    {
      base::ScopedProfile profile(profile_.swiglu_ms);
      SwiGlu(gate, up, activated);
    }
    {
      base::ScopedProfile profile(profile_.ffn_down_proj_ms);
      MatVec(weights.down_proj, activated, projected_ffn);
    }
    {
      base::ScopedProfile profile(profile_.ffn_residual_ms);
      AddInPlace(hidden_state, projected_ffn);
    }
  }

  {
    base::ScopedProfile profile(profile_.final_norm_ms);
    RmsNorm(hidden_state, model_.weights.final_norm, config.rms_norm_eps, norm);
  }

  LlamaForwardResult result;
  result.logits = tensor::Tensor::allocate(base::DataType::kDataTypeFp32,
                                           {config.vocab_size},
                                           base::DeviceType::kDeviceCPU);
  std::vector<float> logits;
  {
    base::ScopedProfile profile(profile_.lm_head_ms);
    MatVec(model_.weights.lm_head, norm, logits);
  }
  CHECK_EQ(static_cast<int32_t>(logits.size()), config.vocab_size);
  std::copy(logits.begin(), logits.end(), result.logits.data<float>());
  {
    base::ScopedProfile profile(profile_.argmax_ms);
    result.next_token = ArgMaxToken(result.logits);
  }

  LOG(INFO) << "finish LLaMA HF one-token forward: next_token="
            << result.next_token;
  return result;
}

void LlamaForwardProfile::Log() const {
  LOG(INFO) << "  Forward Stats:";
  LOG(INFO) << "    forward_calls=" << forward_calls;
  LOG(INFO) << "    Embedding & Attention:";
  LOG(INFO) << "      embedding_ms=" << embedding_ms;
  LOG(INFO) << "      attention_norm_ms=" << attention_norm_ms;
  LOG(INFO) << "      qkv_proj_ms=" << qkv_proj_ms;
  LOG(INFO) << "      rope_ms=" << rope_ms;
  LOG(INFO) << "      kv_cache_ms=" << kv_cache_ms;
  LOG(INFO) << "      attention_ms=" << attention_ms;
  LOG(INFO) << "      attention_output_proj_ms=" << attention_output_proj_ms;
  LOG(INFO) << "      attention_residual_ms=" << attention_residual_ms;
  LOG(INFO) << "    FFN:";
  LOG(INFO) << "      ffn_norm_ms=" << ffn_norm_ms;
  LOG(INFO) << "      ffn_up_gate_proj_ms=" << ffn_up_gate_proj_ms;
  LOG(INFO) << "      swiglu_ms=" << swiglu_ms;
  LOG(INFO) << "      ffn_down_proj_ms=" << ffn_down_proj_ms;
  LOG(INFO) << "      ffn_residual_ms=" << ffn_residual_ms;
  LOG(INFO) << "    Final:";
  LOG(INFO) << "      final_norm_ms=" << final_norm_ms;
  LOG(INFO) << "      lm_head_ms=" << lm_head_ms;
  LOG(INFO) << "      argmax_ms=" << argmax_ms;
}

}  // namespace model
