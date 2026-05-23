#pragma once

#include <absl/status/statusor.h>

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace model {

struct HfLlamaConfig {
  std::vector<std::string> architectures;
  bool attention_bias = false;
  int32_t bos_token_id = -1;
  int32_t eos_token_id = -1;
  std::string hidden_act;
  int32_t hidden_size = 0;
  double initializer_range = 0.0;
  int32_t intermediate_size = 0;
  int32_t max_position_embeddings = 0;
  std::string model_type;
  int32_t num_attention_heads = 0;
  int32_t num_hidden_layers = 0;
  int32_t num_key_value_heads = 0;
  int32_t pretraining_tp = 1;
  double rms_norm_eps = 0.0;
  nlohmann::json rope_scaling;
  double rope_theta = 0.0;
  bool tie_word_embeddings = false;
  std::string torch_dtype;
  std::string transformers_version;
  bool use_cache = true;
  int32_t vocab_size = 0;
};

absl::StatusOr<HfLlamaConfig> LoadHfLlamaConfig(const std::string& model_dir);

}  // namespace model
