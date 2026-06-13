#include "model/llama_hf_forward.h"

#include <absl/status/status.h>
#include <absl/strings/str_cat.h>
#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "base/profile.h"
#include "base/types.h"

namespace model {

LlamaHfRuntime::LlamaHfRuntime(const LlamaHfModel& model,
                               base::DeviceType device_type)
    : model_(model), backend_(CreateLlamaBackend(device_type)) {
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
    backend_->Embedding(model_.weights.token_embedding, token_id, hidden_state);
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
      backend_->RmsNorm(hidden_state, weights.input_layernorm,
                        config.rms_norm_eps, norm);
    }
    {
      base::ScopedProfile profile(profile_.qkv_proj_ms);
      backend_->MatVec(weights.q_proj, norm, query);
      backend_->MatVec(weights.k_proj, norm, key);
      backend_->MatVec(weights.v_proj, norm, value);
    }

    {
      base::ScopedProfile profile(profile_.rope_ms);
      backend_->ApplyRopeToHeads(query, config.num_attention_heads, head_size_,
                                 position, config.rope_theta);
      backend_->ApplyRopeToHeads(key, config.num_key_value_heads, head_size_,
                                 position, config.rope_theta);
    }

    {
      base::ScopedProfile profile(profile_.kv_cache_ms);
      backend_->StoreKvCache(
          key, value, position, config.max_position_embeddings, kv_dim_,
          layer_caches_[layer].key, layer_caches_[layer].value);
    }
    {
      base::ScopedProfile profile(profile_.attention_ms);
      backend_->AttentionWithCache(query, layer_caches_[layer].key,
                                   layer_caches_[layer].value, position,
                                   config.num_attention_heads, head_size_,
                                   kv_dim_, kv_mul_, attention_output);
    }
    {
      base::ScopedProfile profile(profile_.attention_output_proj_ms);
      backend_->MatVec(weights.o_proj, attention_output, projected_attention);
    }
    {
      base::ScopedProfile profile(profile_.attention_residual_ms);
      backend_->AddInPlace(hidden_state, projected_attention);
    }

    {
      base::ScopedProfile profile(profile_.ffn_norm_ms);
      backend_->RmsNorm(hidden_state, weights.post_attention_layernorm,
                        config.rms_norm_eps, norm);
    }
    {
      base::ScopedProfile profile(profile_.ffn_up_gate_proj_ms);
      backend_->MatVec(weights.gate_proj, norm, gate);
      backend_->MatVec(weights.up_proj, norm, up);
    }
    {
      base::ScopedProfile profile(profile_.swiglu_ms);
      backend_->SwiGlu(gate, up, activated);
    }
    {
      base::ScopedProfile profile(profile_.ffn_down_proj_ms);
      backend_->MatVec(weights.down_proj, activated, projected_ffn);
    }
    {
      base::ScopedProfile profile(profile_.ffn_residual_ms);
      backend_->AddInPlace(hidden_state, projected_ffn);
    }
  }

  {
    base::ScopedProfile profile(profile_.final_norm_ms);
    backend_->RmsNorm(hidden_state, model_.weights.final_norm,
                      config.rms_norm_eps, norm);
  }

  LlamaForwardResult result;
  result.logits = tensor::Tensor::allocate(base::DataType::kDataTypeFp32,
                                           {config.vocab_size},
                                           base::DeviceType::kDeviceCPU);
  std::vector<float> logits;
  {
    base::ScopedProfile profile(profile_.lm_head_ms);
    backend_->MatVec(model_.weights.lm_head, norm, logits);
  }
  CHECK_EQ(static_cast<int32_t>(logits.size()), config.vocab_size);
  std::copy(logits.begin(), logits.end(), result.logits.data<float>());
  {
    base::ScopedProfile profile(profile_.argmax_ms);
    result.next_token = backend_->ArgMaxToken(result.logits);
  }

  LOG(INFO) << "finish LLaMA HF one-token forward: next_token="
            << result.next_token;
  return result;
}

void LlamaForwardProfile::Log() const {
  LOG(INFO) << "  Forward Stats:";
  LOG(INFO) << "    forward_calls=" << forward_calls;
  LOG(INFO) << "    Embedding & Attention:";
  LOG(INFO) << "      embedding_s=" << embedding_ms / 1000.0;
  LOG(INFO) << "      attention_norm_s=" << attention_norm_ms / 1000.0;
  LOG(INFO) << "      qkv_proj_s=" << qkv_proj_ms / 1000.0;
  LOG(INFO) << "      rope_s=" << rope_ms / 1000.0;
  LOG(INFO) << "      kv_cache_s=" << kv_cache_ms / 1000.0;
  LOG(INFO) << "      attention_s=" << attention_ms / 1000.0;
  LOG(INFO) << "      attention_output_proj_s="
            << attention_output_proj_ms / 1000.0;
  LOG(INFO) << "      attention_residual_s=" << attention_residual_ms / 1000.0;
  LOG(INFO) << "    FFN:";
  LOG(INFO) << "      ffn_norm_s=" << ffn_norm_ms / 1000.0;
  LOG(INFO) << "      ffn_up_gate_proj_s=" << ffn_up_gate_proj_ms / 1000.0;
  LOG(INFO) << "      swiglu_s=" << swiglu_ms / 1000.0;
  LOG(INFO) << "      ffn_down_proj_s=" << ffn_down_proj_ms / 1000.0;
  LOG(INFO) << "      ffn_residual_s=" << ffn_residual_ms / 1000.0;
  LOG(INFO) << "    Final:";
  LOG(INFO) << "      final_norm_s=" << final_norm_ms / 1000.0;
  LOG(INFO) << "      lm_head_s=" << lm_head_ms / 1000.0;
  LOG(INFO) << "      argmax_s=" << argmax_ms / 1000.0;
}

}  // namespace model
