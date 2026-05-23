#pragma once

#include <absl/status/statusor.h>

#include <cstdint>
#include <string>

namespace model {

struct HfLlamaConfig {
  std::string architecture;
  int32_t hidden_size = 0;
  int32_t intermediate_size = 0;
  int32_t max_position_embeddings = 0;
  std::string model_type;
  int32_t num_hidden_layers = 0;
  int32_t num_attention_heads = 0;
  int32_t num_key_value_heads = 0;
  double rms_norm_eps = 0.0;
  double rope_theta = 0.0;
  bool tie_word_embeddings = false;
  std::string torch_dtype;
  int32_t vocab_size = 0;
};

absl::StatusOr<HfLlamaConfig> LoadHfLlamaConfig(const std::string& model_dir);

}  // namespace model
