#pragma once

#include <absl/status/statusor.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "model/llama_hf_model_loader.h"
#include "tokenizer/tokenizer.h"

namespace runtime {

struct GenerationConfig {
  int32_t max_new_tokens = 32;
  bool trace_steps = false;
  int32_t trace_top_k = 5;
};

struct GenerationStep {
  int32_t step = 0;
  int32_t position = 0;
  int32_t input_token_id = 0;
  int32_t next_token_id = 0;
  std::vector<std::pair<int32_t, float>> top_logits;
};

struct GenerationProfile {
  size_t prompt_tokens = 0;
  size_t generated_tokens = 0;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  double total_ms = 0.0;
};

struct GenerationResult {
  std::string text;
  std::vector<int32_t> prompt_tokens;
  std::vector<int32_t> tokens;
  std::vector<GenerationStep> steps;
  GenerationProfile profile;
};

absl::StatusOr<GenerationResult> GenerateText(
    const model::LlamaHfModel& model, const tokenizer::Tokenizer& tokenizer,
    const std::string& prompt, const GenerationConfig& config);

}  // namespace runtime
