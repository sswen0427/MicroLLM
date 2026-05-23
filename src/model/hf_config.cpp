#include "model/hf_config.h"

#include <absl/status/status.h>
#include <absl/strings/str_cat.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace model {

namespace {

int32_t ReadOptionalInt(const nlohmann::json& json, const std::string& key,
                        int32_t default_value) {
  if (!json.contains(key) || json.at(key).is_null()) {
    return default_value;
  }
  return json.at(key).get<int32_t>();
}

}  // namespace

void from_json(const nlohmann::json& json, HfLlamaConfig& config) {
  config.raw_json = json;
  config.name_or_path = json.value("_name_or_path", "");
  config.architectures =
      json.value("architectures", std::vector<std::string>{});
  config.attention_bias = json.value("attention_bias", false);
  config.attention_dropout = json.value("attention_dropout", 0.0);
  config.bos_token_id = ReadOptionalInt(json, "bos_token_id", -1);
  config.eos_token_id = ReadOptionalInt(json, "eos_token_id", -1);
  config.hidden_act = json.value("hidden_act", "");
  config.hidden_size = json.at("hidden_size").get<int32_t>();
  config.initializer_range = json.value("initializer_range", 0.0);
  config.intermediate_size = json.at("intermediate_size").get<int32_t>();
  config.max_position_embeddings = json.value("max_position_embeddings", 0);
  config.model_type = json.at("model_type").get<std::string>();
  config.mlp_bias = json.value("mlp_bias", false);
  config.num_hidden_layers = json.at("num_hidden_layers").get<int32_t>();
  config.num_attention_heads = json.at("num_attention_heads").get<int32_t>();
  config.num_key_value_heads =
      json.value("num_key_value_heads", config.num_attention_heads);
  config.pad_token_id = ReadOptionalInt(json, "pad_token_id", -1);
  config.pretraining_tp = json.value("pretraining_tp", 1);
  config.rms_norm_eps = json.value("rms_norm_eps", 0.0);
  config.rope_scaling = json.value("rope_scaling", nlohmann::json(nullptr));
  config.rope_theta = json.value("rope_theta", 0.0);
  config.tie_word_embeddings = json.value("tie_word_embeddings", false);
  config.torch_dtype = json.value("torch_dtype", "");
  config.transformers_version = json.value("transformers_version", "");
  config.use_cache = json.value("use_cache", true);
  config.vocab_size = json.at("vocab_size").get<int32_t>();
}

std::string HfLlamaConfig::architecture() const {
  if (architectures.empty()) {
    return "";
  }
  return architectures.front();
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
  if (config.attention_bias) {
    return absl::InvalidArgumentError(
        "attention_bias=true is not supported yet.");
  }
  if (config.hidden_act != "silu") {
    return absl::InvalidArgumentError(
        absl::StrCat("Only hidden_act=silu is supported, got: ",
                     config.hidden_act));
  }
  return config;
}

}  // namespace model
