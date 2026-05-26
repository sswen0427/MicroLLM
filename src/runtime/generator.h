#pragma once

#include <absl/status/statusor.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "model/llama_hf_model_loader.h"
#include "tokenizer/tokenizer.h"

namespace runtime {

struct GenerationConfig {
  int32_t max_new_tokens = 32;
  int32_t trace_top_k = 5;
};

struct GenerationStep {
  int32_t step = 0;
  int32_t position = 0;
  int32_t input_token_id = 0;
  int32_t next_token_id = 0;
  std::vector<std::pair<int32_t, float>> top_logits;
};

struct GenerationResult {
  std::string text;
  std::vector<int32_t> prompt_tokens;
  std::vector<int32_t> tokens;
  std::vector<GenerationStep> steps;
};

absl::StatusOr<GenerationResult> GenerateText(
    const model::LlamaHfModel& model, const tokenizer::Tokenizer& tokenizer,
    const std::string& prompt, const GenerationConfig& config);

}  // namespace runtime
