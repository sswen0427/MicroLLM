#include "model/llama_hf_model_loader.h"

#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_cat.h>
#include <glog/logging.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "model/llama_safetensors_loader.h"
#include "model/llama_tensor_names.h"

namespace model {
namespace {

absl::StatusOr<std::string> FindSingleSafetensorsFile(
    const std::string& model_dir) {
  const std::filesystem::path dir(model_dir);
  if (!std::filesystem::exists(dir)) {
    return absl::NotFoundError(
        absl::StrCat("Model directory does not exist: ", model_dir));
  }
  if (!std::filesystem::is_directory(dir)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Model path is not a directory: ", model_dir));
  }

  const std::filesystem::path preferred = dir / "model.safetensors";
  if (std::filesystem::exists(preferred)) {
    return preferred.string();
  }

  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());

  if (files.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No .safetensors file found in: ", model_dir));
  }
  if (files.size() > 1) {
    return absl::UnimplementedError(
        absl::StrCat("Multiple .safetensors files found in ", model_dir,
                     ". Sharded safetensors are not supported yet."));
  }
  return files.front().string();
}

absl::Status ValidateSupportedLlamaConfig(const HfLlamaConfig& config) {
  if (config.model_type != "llama") {
    return absl::UnimplementedError(absl::StrCat(
        "Only llama model_type is supported, got ", config.model_type));
  }
  if (config.hidden_size <= 0 || config.intermediate_size <= 0 ||
      config.num_hidden_layers <= 0 || config.num_attention_heads <= 0 ||
      config.num_key_value_heads <= 0 || config.vocab_size <= 0) {
    return absl::InvalidArgumentError(
        "Invalid non-positive LLaMA config value.");
  }
  if (config.hidden_size % config.num_attention_heads != 0) {
    return absl::InvalidArgumentError(
        "hidden_size must be divisible by num_attention_heads.");
  }
  if (config.attention_bias) {
    return absl::UnimplementedError(
        "attention_bias=true is not supported yet.");
  }
  if (config.hidden_act != "silu") {
    return absl::UnimplementedError(absl::StrCat(
        "Only hidden_act=silu is supported, got: ", config.hidden_act));
  }
  return absl::OkStatus();
}

absl::StatusOr<tensor::Tensor> LoadRequiredTensor(
    const LlamaSafetensorsLoader& loader, const std::string& tensor_name) {
  auto tensor_or = loader.LoadTensor(tensor_name);
  if (!tensor_or.ok()) {
    return tensor_or.status();
  }
  return std::move(*tensor_or);
}

absl::StatusOr<LlamaHfLayerWeights> LoadLayerWeights(
    const LlamaSafetensorsLoader& loader, int32_t layer) {
  LlamaHfLayerWeights weights;

  auto input_layernorm = LoadRequiredTensor(
      loader, LlamaLayerTensorName(layer, LlamaTensorKind::kInputLayerNorm));
  if (!input_layernorm.ok()) {
    return input_layernorm.status();
  }
  weights.input_layernorm = std::move(*input_layernorm);

  auto post_attention_layernorm = LoadRequiredTensor(
      loader,
      LlamaLayerTensorName(layer, LlamaTensorKind::kPostAttentionLayerNorm));
  if (!post_attention_layernorm.ok()) {
    return post_attention_layernorm.status();
  }
  weights.post_attention_layernorm = std::move(*post_attention_layernorm);

  auto q_proj = LoadRequiredTensor(
      loader, LlamaLayerTensorName(layer, LlamaTensorKind::kQProj));
  if (!q_proj.ok()) {
    return q_proj.status();
  }
  weights.q_proj = std::move(*q_proj);

  auto k_proj = LoadRequiredTensor(
      loader, LlamaLayerTensorName(layer, LlamaTensorKind::kKProj));
  if (!k_proj.ok()) {
    return k_proj.status();
  }
  weights.k_proj = std::move(*k_proj);

  auto v_proj = LoadRequiredTensor(
      loader, LlamaLayerTensorName(layer, LlamaTensorKind::kVProj));
  if (!v_proj.ok()) {
    return v_proj.status();
  }
  weights.v_proj = std::move(*v_proj);

  auto o_proj = LoadRequiredTensor(
      loader, LlamaLayerTensorName(layer, LlamaTensorKind::kOProj));
  if (!o_proj.ok()) {
    return o_proj.status();
  }
  weights.o_proj = std::move(*o_proj);

  auto gate_proj = LoadRequiredTensor(
      loader, LlamaLayerTensorName(layer, LlamaTensorKind::kGateProj));
  if (!gate_proj.ok()) {
    return gate_proj.status();
  }
  weights.gate_proj = std::move(*gate_proj);

  auto up_proj = LoadRequiredTensor(
      loader, LlamaLayerTensorName(layer, LlamaTensorKind::kUpProj));
  if (!up_proj.ok()) {
    return up_proj.status();
  }
  weights.up_proj = std::move(*up_proj);

  auto down_proj = LoadRequiredTensor(
      loader, LlamaLayerTensorName(layer, LlamaTensorKind::kDownProj));
  if (!down_proj.ok()) {
    return down_proj.status();
  }
  weights.down_proj = std::move(*down_proj);

  return weights;
}

absl::StatusOr<LlamaHfModelWeights> LoadWeights(
    const LlamaSafetensorsLoader& loader, const HfLlamaConfig& config) {
  LlamaHfModelWeights weights;

  auto token_embedding = LoadRequiredTensor(
      loader, LlamaTensorName(LlamaTensorKind::kTokenEmbedding));
  if (!token_embedding.ok()) {
    return token_embedding.status();
  }
  weights.token_embedding = std::move(*token_embedding);

  auto final_norm =
      LoadRequiredTensor(loader, LlamaTensorName(LlamaTensorKind::kFinalNorm));
  if (!final_norm.ok()) {
    return final_norm.status();
  }
  weights.final_norm = std::move(*final_norm);

  auto lm_head =
      LoadRequiredTensor(loader, LlamaTensorName(LlamaTensorKind::kLmHead));
  if (!lm_head.ok()) {
    return lm_head.status();
  }
  weights.lm_head = std::move(*lm_head);

  weights.layers.reserve(config.num_hidden_layers);
  for (int32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    auto layer_weights = LoadLayerWeights(loader, layer);
    if (!layer_weights.ok()) {
      return layer_weights.status();
    }
    weights.layers.push_back(std::move(*layer_weights));
  }

  return weights;
}

void LogLlamaConfig(const HfLlamaConfig& config) {
  LOG(INFO) << "architectures[0]: "
            << (config.architectures.empty() ? "" : config.architectures[0]);
  LOG(INFO) << "attention_bias: " << config.attention_bias;
  LOG(INFO) << "bos_token_id: " << config.bos_token_id;
  LOG(INFO) << "eos_token_id: " << config.eos_token_id;
  LOG(INFO) << "hidden_act: " << config.hidden_act;
  LOG(INFO) << "hidden_size: " << config.hidden_size;
  LOG(INFO) << "initializer_range: " << config.initializer_range;
  LOG(INFO) << "intermediate_size: " << config.intermediate_size;
  LOG(INFO) << "max_position_embeddings: " << config.max_position_embeddings;
  LOG(INFO) << "model_type: " << config.model_type;
  LOG(INFO) << "attention_heads: " << config.num_attention_heads;
  LOG(INFO) << "layers: " << config.num_hidden_layers;
  LOG(INFO) << "kv_heads: " << config.num_key_value_heads;
  LOG(INFO) << "pretraining_tp: " << config.pretraining_tp;
  LOG(INFO) << "rms_norm_eps: " << config.rms_norm_eps;
  LOG(INFO) << "rope_scaling: " << config.rope_scaling.dump();
  LOG(INFO) << "rope_theta: " << config.rope_theta;
  LOG(INFO) << "tie_word_embeddings: " << config.tie_word_embeddings;
  LOG(INFO) << "torch_dtype: " << config.torch_dtype;
  LOG(INFO) << "transformers_version: " << config.transformers_version;
  LOG(INFO) << "use_cache: " << config.use_cache;
  LOG(INFO) << "vocab_size: " << config.vocab_size;
}

}  // namespace

absl::StatusOr<std::unique_ptr<LlamaHfModel>> LoadLlamaHfModel(
    const std::string& model_dir) {
  auto config_or = LoadHfLlamaConfig(model_dir);
  if (!config_or.ok()) {
    return config_or.status();
  }
  const HfLlamaConfig& config = *config_or;
  const absl::Status config_status = ValidateSupportedLlamaConfig(config);
  if (!config_status.ok()) {
    return config_status;
  }
  LogLlamaConfig(config);

  auto safetensors_path_or = FindSingleSafetensorsFile(model_dir);
  if (!safetensors_path_or.ok()) {
    return safetensors_path_or.status();
  }
  LOG(INFO) << "safetensors: " << *safetensors_path_or;

  auto loader_or = LlamaSafetensorsLoader::Open(*safetensors_path_or);
  if (!loader_or.ok()) {
    return loader_or.status();
  }
  const auto& loader = **loader_or;
  LOG(INFO) << "tensor_count: " << loader.TensorCount();

  auto weights_or = LoadWeights(loader, config);
  if (!weights_or.ok()) {
    return weights_or.status();
  }

  auto model = std::make_unique<LlamaHfModel>();
  model->config = config;
  model->weights = std::move(*weights_or);

  LOG(INFO) << "LLaMA safetensors weights loaded: layers="
            << model->weights.layers.size();
  return std::move(model);
}

}  // namespace model
