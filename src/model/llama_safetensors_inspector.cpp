#include "model/llama_safetensors_inspector.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "io/safetensors_reader.h"

namespace model {
namespace {

struct ExpectedTensor {
  std::string name;
  std::vector<std::size_t> shape;
};

std::string FormatShape(const std::vector<std::size_t>& shape) {
  std::string text = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      text += ", ";
    }
    text += std::to_string(shape[i]);
  }
  text += "]";
  return text;
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
    if (entry.is_regular_file() &&
        entry.path().extension() == ".safetensors") {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());

  if (files.empty()) {
    return absl::NotFoundError(
        absl::StrCat("No .safetensors file found in: ", model_dir));
  }
  if (files.size() > 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Multiple .safetensors files found in ", model_dir,
                     ". Sharded safetensors are not supported yet."));
  }
  return files.front().string();
}

std::vector<ExpectedTensor> BuildExpectedTensors(const HfLlamaConfig& config) {
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t head_dim =
      config.hidden_size / config.num_attention_heads;
  const std::size_t kv_dim = config.num_key_value_heads * head_dim;

  std::vector<ExpectedTensor> expected;
  expected.push_back({"model.embed_tokens.weight", {vocab_size, hidden_size}});
  expected.push_back({"model.norm.weight", {hidden_size}});
  if (!config.tie_word_embeddings) {
    expected.push_back({"lm_head.weight", {vocab_size, hidden_size}});
  }

  for (int32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    const std::string prefix =
        absl::StrCat("model.layers.", layer, ".");
    expected.push_back(
        {prefix + "input_layernorm.weight", {hidden_size}});
    expected.push_back(
        {prefix + "post_attention_layernorm.weight", {hidden_size}});
    expected.push_back(
        {prefix + "self_attn.q_proj.weight", {hidden_size, hidden_size}});
    expected.push_back(
        {prefix + "self_attn.k_proj.weight", {kv_dim, hidden_size}});
    expected.push_back(
        {prefix + "self_attn.v_proj.weight", {kv_dim, hidden_size}});
    expected.push_back(
        {prefix + "self_attn.o_proj.weight", {hidden_size, hidden_size}});
    expected.push_back(
        {prefix + "mlp.gate_proj.weight", {intermediate_size, hidden_size}});
    expected.push_back(
        {prefix + "mlp.up_proj.weight", {intermediate_size, hidden_size}});
    expected.push_back(
        {prefix + "mlp.down_proj.weight", {hidden_size, intermediate_size}});
  }
  return expected;
}

absl::Status PrintAndValidateTensor(const io::SafetensorsReader& reader,
                                    const ExpectedTensor& expected,
                                    std::ostream& output) {
  auto info_or = reader.tensor_info(expected.name);
  if (!info_or.ok()) {
    return info_or.status();
  }

  const auto& info = *info_or;
  output << "  " << expected.name << " " << FormatShape(info.shape) << " "
         << info.dtype << " bytes=" << info.byte_size << "\n";
  if (info.shape != expected.shape) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected shape for ", expected.name, ": got ",
                     FormatShape(info.shape), ", expected ",
                     FormatShape(expected.shape)));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status InspectLlamaSafetensorsModel(const std::string& model_dir,
                                          std::ostream& output) {
  auto config_or = LoadHfLlamaConfig(model_dir);
  if (!config_or.ok()) {
    return config_or.status();
  }
  auto safetensors_path_or = FindSingleSafetensorsFile(model_dir);
  if (!safetensors_path_or.ok()) {
    return safetensors_path_or.status();
  }
  return InspectLlamaSafetensorsModel(*config_or, *safetensors_path_or, output);
}

absl::Status InspectLlamaSafetensorsModel(const HfLlamaConfig& config,
                                          const std::string& safetensors_path,
                                          std::ostream& output) {
  if (config.model_type != "llama") {
    return absl::InvalidArgumentError(
        absl::StrCat("Only llama model_type is supported by this inspector, got ",
                     config.model_type));
  }

  auto reader_or = io::SafetensorsReader::Open(safetensors_path);
  if (!reader_or.ok()) {
    return reader_or.status();
  }
  const auto& reader = **reader_or;

  output << "model_type: " << config.model_type << "\n";
  output << "architecture: " << config.architecture << "\n";
  output << "torch_dtype: " << config.torch_dtype << "\n";
  output << "safetensors: " << safetensors_path << "\n";
  output << "tensor_count: " << reader.tensor_count() << "\n";
  output << "layers: " << config.num_hidden_layers << "\n";
  output << "hidden_size: " << config.hidden_size << "\n";
  output << "intermediate_size: " << config.intermediate_size << "\n";
  output << "attention_heads: " << config.num_attention_heads << "\n";
  output << "kv_heads: " << config.num_key_value_heads << "\n";
  output << "vocab_size: " << config.vocab_size << "\n";
  output << "tie_word_embeddings: " << config.tie_word_embeddings << "\n";
  output << "max_position_embeddings: " << config.max_position_embeddings
         << "\n";
  output << "rms_norm_eps: " << config.rms_norm_eps << "\n";
  output << "rope_theta: " << config.rope_theta << "\n";
  output << "\nvalidated_tensors:\n";

  for (const auto& expected : BuildExpectedTensors(config)) {
    const absl::Status status =
        PrintAndValidateTensor(reader, expected, output);
    if (!status.ok()) {
      return status;
    }
  }

  output << "\nstatus: ok\n";
  return absl::OkStatus();
}

}  // namespace model
