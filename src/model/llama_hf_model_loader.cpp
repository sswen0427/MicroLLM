#include "model/llama_hf_model_loader.h"

#include <absl/status/statusor.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <glog/logging.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "base/types.h"
#include "model/hf_config.h"
#include "model/llama_safetensors_loader.h"
#include "model/llama_tensor_names.h"
#include "tensor/tensor.h"

namespace model {
namespace {

struct TensorToLoad {
  std::string name;
};

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

std::string DataTypeName(base::DataType data_type) {
  switch (data_type) {
    case base::DataType::kDataTypeFp32:
      return "fp32";
    case base::DataType::kDataTypeFp16:
      return "fp16";
    case base::DataType::kDataTypeBf16:
      return "bf16";
    case base::DataType::kDataTypeInt8:
      return "int8";
    case base::DataType::kDataTypeInt32:
      return "int32";
    default:
      return "unknown";
  }
}

std::vector<int32_t> TensorShape(const tensor::Tensor& tensor) {
  std::vector<int32_t> shape;
  shape.reserve(tensor.dims_size());
  for (int32_t i = 0; i < tensor.dims_size(); ++i) {
    shape.push_back(tensor.get_dim(i));
  }
  return shape;
}

std::vector<TensorToLoad> BuildInitialLlamaTensorList() {
  return {
      {LlamaTensorName(LlamaTensorKind::kTokenEmbedding)},
      {LlamaTensorName(LlamaTensorKind::kFinalNorm)},
      {LlamaTensorName(LlamaTensorKind::kLmHead)},
      {LlamaLayerTensorName(0, LlamaTensorKind::kInputLayerNorm)},
      {LlamaLayerTensorName(0, LlamaTensorKind::kPostAttentionLayerNorm)},
      {LlamaLayerTensorName(0, LlamaTensorKind::kQProj)},
      {LlamaLayerTensorName(0, LlamaTensorKind::kKProj)},
      {LlamaLayerTensorName(0, LlamaTensorKind::kVProj)},
      {LlamaLayerTensorName(0, LlamaTensorKind::kOProj)},
      {LlamaLayerTensorName(0, LlamaTensorKind::kGateProj)},
      {LlamaLayerTensorName(0, LlamaTensorKind::kUpProj)},
      {LlamaLayerTensorName(0, LlamaTensorKind::kDownProj)},
  };
}

absl::Status LoadAndLogTensor(const LlamaSafetensorsLoader& loader,
                              const std::string& tensor_name) {
  auto tensor_or = loader.LoadTensor(tensor_name);
  if (!tensor_or.ok()) {
    return tensor_or.status();
  }
  const tensor::Tensor& tensor = *tensor_or;
  LOG(INFO) << "loaded tensor: name=" << tensor_name
            << ", dtype=" << DataTypeName(tensor.data_type())
            << ", shape=" << absl::StrJoin(TensorShape(tensor), "x")
            << ", bytes=" << tensor.byte_size();
  return absl::OkStatus();
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

absl::Status LoadLlamaHfModel(const std::string& model_dir) {
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

  for (const TensorToLoad& tensor : BuildInitialLlamaTensorList()) {
    const absl::Status status = LoadAndLogTensor(loader, tensor.name);
    if (!status.ok()) {
      return status;
    }
  }

  LOG(INFO) << "initial LLaMA safetensors weights loaded";
  return absl::OkStatus();
}

}  // namespace model
