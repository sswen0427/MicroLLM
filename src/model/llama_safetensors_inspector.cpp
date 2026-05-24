#include "model/llama_safetensors_inspector.h"

#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <glog/logging.h>

#include <algorithm>
#include <filesystem>
#include <safetensors.hh>
#include <string>
#include <utility>
#include <vector>

#include "model/hf_config.h"
#include "model/llama_tensor_names.h"

namespace model {
namespace {

absl::StatusOr<safetensors::safetensors_t> LoadSafetensorsFile(
    const std::string& safetensors_path) {
  safetensors::safetensors_t safetensors;
  std::string warn;
  std::string err;
  if (!safetensors::mmap_from_file(safetensors_path, &safetensors, &warn,
                                   &err)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to open safetensors file: ", safetensors_path,
                     err.empty() ? "" : absl::StrCat(", error: ", err)));
  }
  if (!warn.empty()) {
    LOG(WARNING) << "safetensors warning for " << safetensors_path << ": "
                 << warn;
  }

  // File-level integrity is delegated to safetensors-cpp.
  std::string offset_error;
  if (!safetensors::validate_data_offsets(safetensors, offset_error)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid safetensors data offsets in ", safetensors_path,
        offset_error.empty() ? "" : absl::StrCat(", error: ", offset_error)));
  }
  return safetensors;
}

absl::Status LogSafetensorsTensorMetadata(
    const safetensors::safetensors_t& safetensors,
    const std::string& tensor_name) {
  safetensors::tensor_t tensor;
  if (!safetensors.tensors.at(tensor_name, &tensor)) {
    return absl::NotFoundError(
        absl::StrCat("Tensor not found in safetensors: ", tensor_name));
  }
  if (tensor.data_offsets.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid data offsets for tensor: ", tensor_name));
  }
  const size_t begin = tensor.data_offsets[0];
  const size_t end = tensor.data_offsets[1];
  if (begin > end) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid data offsets for tensor ", tensor_name, ": [",
                     begin, ", ", end, "]"));
  }

  LOG(INFO) << "tensor: name=" << tensor_name
            << ", dtype=" << safetensors::get_dtype_str(tensor.dtype)
            << ", shape=" << absl::StrJoin(tensor.shape, "x")
            << ", bytes=" << end - begin;
  return absl::OkStatus();
}

absl::Status ValidateSupportedLlamaConfig(const HfLlamaConfig& config) {
  if (config.model_type != "llama") {
    return absl::UnimplementedError(absl::StrCat(
        "Only llama model_type is supported by this inspector, got ",
        config.model_type));
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

absl::Status InspectLlamaSafetensorsFile(const HfLlamaConfig& config,
                                         const std::string& safetensors_path) {
  const absl::Status config_status = ValidateSupportedLlamaConfig(config);
  if (!config_status.ok()) {
    return config_status;
  }

  auto safetensors_or = LoadSafetensorsFile(safetensors_path);
  if (!safetensors_or.ok()) {
    return safetensors_or.status();
  }
  const auto& safetensors = *safetensors_or;

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
  LOG(INFO) << "safetensors: " << safetensors_path;
  LOG(INFO) << "tensor_count: " << safetensors.tensors.size();

  const absl::Status token_embedding_status = LogSafetensorsTensorMetadata(
      safetensors, LlamaTensorName(LlamaTensorKind::kTokenEmbedding));
  if (!token_embedding_status.ok()) {
    return token_embedding_status;
  }
  const absl::Status layer_q_status = LogSafetensorsTensorMetadata(
      safetensors, LlamaLayerTensorName(0, LlamaTensorKind::kQProj));
  if (!layer_q_status.ok()) {
    return layer_q_status;
  }
  const absl::Status layer_gate_status = LogSafetensorsTensorMetadata(
      safetensors, LlamaLayerTensorName(0, LlamaTensorKind::kGateProj));
  if (!layer_gate_status.ok()) {
    return layer_gate_status;
  }
  const absl::Status final_norm_status = LogSafetensorsTensorMetadata(
      safetensors, LlamaTensorName(LlamaTensorKind::kFinalNorm));
  if (!final_norm_status.ok()) {
    return final_norm_status;
  }
  const absl::Status lm_head_status = LogSafetensorsTensorMetadata(
      safetensors, LlamaTensorName(LlamaTensorKind::kLmHead));
  if (!lm_head_status.ok()) {
    return lm_head_status;
  }

  LOG(INFO) << "safetensors model inspection finished";
  return absl::OkStatus();
}

}  // namespace

absl::Status InspectLlamaSafetensorsModel(const std::string& model_dir) {
  auto config_or = LoadHfLlamaConfig(model_dir);
  if (!config_or.ok()) {
    return config_or.status();
  }
  auto safetensors_path_or = FindSingleSafetensorsFile(model_dir);
  if (!safetensors_path_or.ok()) {
    return safetensors_path_or.status();
  }
  return InspectLlamaSafetensorsFile(*config_or, *safetensors_path_or);
}

}  // namespace model
