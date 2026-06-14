#include "model/llama_backend.h"

#include <glog/logging.h>

#include <cstddef>
#include <memory>

namespace model {

std::unique_ptr<LlamaBackend> CreateCpuLlamaBackend();
std::unique_ptr<LlamaBackend> CreateCudaLlamaBackend();

std::unique_ptr<LlamaBackend> CreateLlamaBackend(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return CreateCpuLlamaBackend();
  }
  if (device_type == base::DeviceType::kDeviceCUDA) {
    return CreateCudaLlamaBackend();
  }
  LOG(FATAL) << "Unsupported LLaMA backend device type: "
             << static_cast<int>(device_type);
  return nullptr;
}

LlamaForwardState CreateLlamaForwardState(const HfLlamaConfig &config) {
  LlamaForwardState state;
  if (config.num_attention_heads > 0) {
    state.head_size = config.hidden_size / config.num_attention_heads;
  }
  state.kv_dim = config.num_key_value_heads * state.head_size;
  if (config.num_key_value_heads > 0) {
    state.kv_mul = config.num_attention_heads / config.num_key_value_heads;
  }

  if (config.num_hidden_layers <= 0 || config.max_position_embeddings <= 0 ||
      state.kv_dim <= 0) {
    return state;
  }

  state.layer_caches.resize(config.num_hidden_layers);
  const size_t cache_size =
      static_cast<size_t>(config.max_position_embeddings) * state.kv_dim;
  for (LlamaLayerCache &cache : state.layer_caches) {
    cache.key.assign(cache_size, 0.0f);
    cache.value.assign(cache_size, 0.0f);
  }
  return state;
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
