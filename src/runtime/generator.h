#pragma once

#include <absl/status/statusor.h>

#include <cstdint>
#include <string>
#include <vector>

#include "model/llama_hf_model_loader.h"
#include "op/encode.h"

namespace runtime {

struct GenerationConfig {
  int32_t max_new_tokens = 32;
};

struct GenerationResult {
  std::string text;
  std::vector<int32_t> tokens;
};

absl::StatusOr<GenerationResult> GenerateText(
    const model::LlamaHfModel& model, const op::EncodeLayerBase& tokenizer,
    const std::string& prompt, const GenerationConfig& config);

}  // namespace runtime
