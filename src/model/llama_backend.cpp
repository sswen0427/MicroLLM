#include "model/llama_backend.h"

#include <absl/status/status.h>
#include <absl/strings/str_cat.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "base/profile.h"
#include "base/types.h"
#include "model/llama_cpu_backend.h"
#include "model/llama_cuda_backend.h"

namespace model {
namespace {

std::vector<float> CopyTensorToVector(tensor::Tensor tensor) {
  tensor.to_cpu();
  std::vector<float> values(tensor.size());
  std::copy(tensor.data<float>(), tensor.data<float>() + tensor.size(),
            values.begin());
  return values;
}

tensor::Tensor CopyVectorToBackendTensor(const std::vector<float> &values,
                                         base::DeviceType device_type) {
  tensor::Tensor tensor = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {static_cast<int32_t>(values.size())},
      base::DeviceType::kDeviceCPU);
  std::copy(values.begin(), values.end(), tensor.data<float>());
  if (device_type == base::DeviceType::kDeviceCUDA) {
    tensor.to_cuda();
  }
  return tensor;
}

} // namespace

std::unique_ptr<LlamaBackend> CreateLlamaBackend(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return std::make_unique<CpuLlamaBackend>();
  }
  if (device_type == base::DeviceType::kDeviceCUDA) {
    return std::make_unique<CudaLlamaBackend>();
  }
  LOG(FATAL) << "Unsupported LLaMA backend device type: "
             << static_cast<int>(device_type);
  return nullptr;
}

absl::StatusOr<LlamaForwardResult>
LlamaBackend::ForwardToken(const LlamaHfModel &model, LlamaForwardState &state,
                           int32_t token_id, int32_t position) const {
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

  LOG(INFO) << "start LLaMA HF one-token forward: token_id=" << token_id
            << ", position=" << position;
  state.profile.forward_calls += 1;

  tensor::Tensor hidden_state;
  {
    base::ScopedProfile profile(state.profile.embedding_ms);
    hidden_state = EmbeddingTensor(model.weights.token_embedding, token_id);
  }

  tensor::Tensor norm;
  tensor::Tensor query_tensor;
  tensor::Tensor key_tensor;
  tensor::Tensor value_tensor;
  tensor::Tensor projected_attention;
  tensor::Tensor gate;
  tensor::Tensor up;
  tensor::Tensor activated;
  tensor::Tensor projected_ffn;
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> attention_output;

  for (int32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    const LlamaHfLayerWeights &weights = model.weights.layers[layer];

    {
      base::ScopedProfile profile(state.profile.attention_norm_ms);
      norm = RmsNormTensor(hidden_state, weights.input_layernorm,
                           config.rms_norm_eps);
    }
    {
      base::ScopedProfile profile(state.profile.qkv_proj_ms);
      query_tensor = MatVecTensor(weights.q_proj, norm);
      key_tensor = MatVecTensor(weights.k_proj, norm);
      value_tensor = MatVecTensor(weights.v_proj, norm);
      query = CopyTensorToVector(query_tensor);
      key = CopyTensorToVector(key_tensor);
      value = CopyTensorToVector(value_tensor);
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
      projected_attention = MatVecTensor(
          weights.o_proj,
          CopyVectorToBackendTensor(attention_output, device_type()));
    }
    {
      base::ScopedProfile profile(state.profile.attention_residual_ms);
      AddInPlaceTensor(hidden_state, projected_attention);
    }

    {
      base::ScopedProfile profile(state.profile.ffn_norm_ms);
      norm = RmsNormTensor(hidden_state, weights.post_attention_layernorm,
                           config.rms_norm_eps);
    }
    {
      base::ScopedProfile profile(state.profile.ffn_up_gate_proj_ms);
      gate = MatVecTensor(weights.gate_proj, norm);
      up = MatVecTensor(weights.up_proj, norm);
    }
    {
      base::ScopedProfile profile(state.profile.swiglu_ms);
      activated = SwiGluTensor(gate, up);
    }
    {
      base::ScopedProfile profile(state.profile.ffn_down_proj_ms);
      projected_ffn = MatVecTensor(weights.down_proj, activated);
    }
    {
      base::ScopedProfile profile(state.profile.ffn_residual_ms);
      AddInPlaceTensor(hidden_state, projected_ffn);
    }
  }

  {
    base::ScopedProfile profile(state.profile.final_norm_ms);
    norm = RmsNormTensor(hidden_state, model.weights.final_norm,
                         config.rms_norm_eps);
  }

  LlamaForwardResult result;
  {
    base::ScopedProfile profile(state.profile.lm_head_ms);
    result.logits = MatVecTensor(model.weights.lm_head, norm);
  }
  result.logits.to_cpu();
  CHECK_EQ(static_cast<int32_t>(result.logits.size()), config.vocab_size);
  {
    base::ScopedProfile profile(state.profile.argmax_ms);
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

} // namespace model
