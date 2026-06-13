#include "model/llama_hf_forward.h"

#include <cstddef>

namespace model {

LlamaHfRuntime::LlamaHfRuntime(const LlamaHfModel &model,
                               base::DeviceType device_type)
    : model_(model), backend_(CreateLlamaBackend(device_type)) {
  const HfLlamaConfig &config = model_.config;
  if (config.num_attention_heads > 0) {
    state_.head_size = config.hidden_size / config.num_attention_heads;
  }
  state_.kv_dim = config.num_key_value_heads * state_.head_size;
  if (config.num_key_value_heads > 0) {
    state_.kv_mul = config.num_attention_heads / config.num_key_value_heads;
  }

  if (config.num_hidden_layers <= 0 || config.max_position_embeddings <= 0 ||
      state_.kv_dim <= 0) {
    return;
  }

  state_.layer_caches.resize(config.num_hidden_layers);
  const size_t cache_size =
      static_cast<size_t>(config.max_position_embeddings) * state_.kv_dim;
  for (LlamaLayerCache &cache : state_.layer_caches) {
    cache.key.assign(cache_size, 0.0f);
    cache.value.assign(cache_size, 0.0f);
  }
}

absl::StatusOr<LlamaForwardResult> LlamaHfRuntime::ForwardToken(
    int32_t token_id, int32_t position) {
  return backend_->ForwardToken(model_, state_, token_id, position);
}

}  // namespace model
