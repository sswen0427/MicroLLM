#include "model/hf_config.h"

#include <absl/status/status.h>
#include <absl/strings/str_cat.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace model {

void from_json(const nlohmann::json& json, HfLlamaConfig& config) {
  config.architectures =
      json.at("architectures").get<std::vector<std::string>>();
  config.attention_bias = json.at("attention_bias").get<bool>();
  config.bos_token_id = json.at("bos_token_id").get<int32_t>();
  config.eos_token_id = json.at("eos_token_id").get<int32_t>();
  config.hidden_act = json.at("hidden_act").get<std::string>();
  config.hidden_size = json.at("hidden_size").get<int32_t>();
  config.initializer_range = json.at("initializer_range").get<double>();
  config.intermediate_size = json.at("intermediate_size").get<int32_t>();
  config.max_position_embeddings =
      json.at("max_position_embeddings").get<int32_t>();
  config.model_type = json.at("model_type").get<std::string>();
  config.num_attention_heads = json.at("num_attention_heads").get<int32_t>();
  config.num_hidden_layers = json.at("num_hidden_layers").get<int32_t>();
  config.num_key_value_heads = json.at("num_key_value_heads").get<int32_t>();
  config.pretraining_tp = json.at("pretraining_tp").get<int32_t>();
  config.rms_norm_eps = json.at("rms_norm_eps").get<double>();
  config.rope_scaling = json.at("rope_scaling").get<nlohmann::json>();
  config.rope_theta = json.at("rope_theta").get<double>();
  config.tie_word_embeddings = json.at("tie_word_embeddings").get<bool>();
  config.torch_dtype = json.at("torch_dtype").get<std::string>();
  config.transformers_version =
      json.at("transformers_version").get<std::string>();
  config.use_cache = json.at("use_cache").get<bool>();
  config.vocab_size = json.at("vocab_size").get<int32_t>();
}

absl::StatusOr<HfLlamaConfig> LoadHfLlamaConfig(const std::string& model_dir) {
  const std::filesystem::path config_path =
      std::filesystem::path(model_dir) / "config.json";
  std::ifstream input(config_path);
  if (!input) {
    return absl::NotFoundError(
        absl::StrCat("Failed to open config.json: ", config_path.string()));
  }

  HfLlamaConfig config;
  try {
    nlohmann::json json;
    input >> json;
    config = json.get<HfLlamaConfig>();
  } catch (const nlohmann::json::exception& e) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse HuggingFace config.json: ", e.what()));
  }

  return config;
}

}  // namespace model
