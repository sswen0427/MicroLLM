#include "model/hf_config.h"

#include <absl/status/status.h>
#include <absl/strings/str_cat.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace model {

void from_json(const nlohmann::json& json, HfLlamaConfig& config) {
  if (json.contains("architectures") && json.at("architectures").is_array() &&
      !json.at("architectures").empty()) {
    config.architecture = json.at("architectures").front().get<std::string>();
  }

  config.hidden_size = json.at("hidden_size").get<int32_t>();
  config.intermediate_size = json.at("intermediate_size").get<int32_t>();
  config.max_position_embeddings = json.value("max_position_embeddings", 0);
  config.model_type = json.at("model_type").get<std::string>();
  config.num_attention_heads = json.at("num_attention_heads").get<int32_t>();
  config.num_hidden_layers = json.at("num_hidden_layers").get<int32_t>();
  config.num_key_value_heads =
      json.value("num_key_value_heads", config.num_attention_heads);
  config.rms_norm_eps = json.value("rms_norm_eps", 0.0);
  config.rope_theta = json.value("rope_theta", 0.0);
  config.tie_word_embeddings = json.value("tie_word_embeddings", false);
  config.torch_dtype = json.value("torch_dtype", "");
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

  if (config.hidden_size <= 0 || config.intermediate_size <= 0 ||
      config.num_hidden_layers <= 0 || config.num_attention_heads <= 0 ||
      config.num_key_value_heads <= 0 || config.vocab_size <= 0) {
    return absl::InvalidArgumentError(
        "Invalid non-positive value in config.json.");
  }
  if (config.hidden_size % config.num_attention_heads != 0) {
    return absl::InvalidArgumentError(
        "hidden_size must be divisible by num_attention_heads.");
  }
  return config;
}

}  // namespace model
