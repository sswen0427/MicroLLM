#include "model/hf_config.h"

#include <filesystem>
#include <fstream>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "nlohmann/json.hpp"

namespace model {
namespace {

template <typename T>
absl::StatusOr<T> ReadRequired(const nlohmann::json& json,
                               const std::string& key) {
  if (!json.contains(key)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Missing required config key: ", key));
  }
  return json.at(key).get<T>();
}

}  // namespace

absl::StatusOr<HfLlamaConfig> LoadHfLlamaConfig(const std::string& model_dir) {
  const std::filesystem::path config_path =
      std::filesystem::path(model_dir) / "config.json";
  std::ifstream input(config_path);
  if (!input) {
    return absl::NotFoundError(
        absl::StrCat("Failed to open config.json: ", config_path.string()));
  }

  nlohmann::json json;
  input >> json;

  HfLlamaConfig config;

  auto model_type = ReadRequired<std::string>(json, "model_type");
  if (!model_type.ok()) {
    return model_type.status();
  }
  config.model_type = *model_type;

  if (json.contains("architectures") && json.at("architectures").is_array() &&
      !json.at("architectures").empty()) {
    config.architecture = json.at("architectures").front().get<std::string>();
  }
  if (json.contains("torch_dtype")) {
    config.torch_dtype = json.at("torch_dtype").get<std::string>();
  }

  auto hidden_size = ReadRequired<int32_t>(json, "hidden_size");
  auto intermediate_size = ReadRequired<int32_t>(json, "intermediate_size");
  auto num_hidden_layers = ReadRequired<int32_t>(json, "num_hidden_layers");
  auto num_attention_heads = ReadRequired<int32_t>(json, "num_attention_heads");
  auto vocab_size = ReadRequired<int32_t>(json, "vocab_size");
  if (!hidden_size.ok()) return hidden_size.status();
  if (!intermediate_size.ok()) return intermediate_size.status();
  if (!num_hidden_layers.ok()) return num_hidden_layers.status();
  if (!num_attention_heads.ok()) return num_attention_heads.status();
  if (!vocab_size.ok()) return vocab_size.status();

  config.hidden_size = *hidden_size;
  config.intermediate_size = *intermediate_size;
  config.num_hidden_layers = *num_hidden_layers;
  config.num_attention_heads = *num_attention_heads;
  config.vocab_size = *vocab_size;

  config.num_key_value_heads =
      json.value("num_key_value_heads", config.num_attention_heads);
  config.max_position_embeddings = json.value("max_position_embeddings", 0);
  config.rms_norm_eps = json.value("rms_norm_eps", 0.0);
  config.rope_theta = json.value("rope_theta", 0.0);
  config.tie_word_embeddings = json.value("tie_word_embeddings", false);

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
