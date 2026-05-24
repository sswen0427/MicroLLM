#include "runtime/generator.h"

#include <absl/status/status.h>
#include <absl/strings/str_cat.h>
#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/llama_hf_forward.h"

namespace runtime {

absl::StatusOr<GenerationResult> GenerateText(
    const model::LlamaHfModel& model, const tokenizer::Tokenizer& tokenizer,
    const std::string& prompt, const GenerationConfig& config) {
  if (prompt.empty()) {
    return absl::InvalidArgumentError("prompt must not be empty.");
  }
  if (config.max_new_tokens <= 0) {
    return absl::InvalidArgumentError("max_new_tokens must be greater than 0.");
  }

  const std::vector<int32_t> prompt_tokens = tokenizer.Encode(prompt);
  if (prompt_tokens.empty()) {
    return absl::InvalidArgumentError("prompt produced no tokens.");
  }
  if (prompt_tokens.size() + static_cast<size_t>(config.max_new_tokens) >
      static_cast<size_t>(model.config.max_position_embeddings)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "prompt tokens + max_new_tokens exceeds max_position_embeddings: ",
        prompt_tokens.size(), " + ", config.max_new_tokens, " > ",
        model.config.max_position_embeddings));
  }

  model::LlamaHfRuntime runtime(model);
  int32_t next_token = -1;
  for (int32_t pos = 0; pos < static_cast<int32_t>(prompt_tokens.size());
       ++pos) {
    auto forward_or = runtime.ForwardToken(prompt_tokens[pos], pos);
    if (!forward_or.ok()) {
      return forward_or.status();
    }
    next_token = forward_or->next_token;
  }

  GenerationResult result;
  result.tokens.reserve(config.max_new_tokens);
  int32_t pos = static_cast<int32_t>(prompt_tokens.size());
  for (int32_t step = 0; step < config.max_new_tokens; ++step, ++pos) {
    if (next_token < 0 || tokenizer.IsEndToken(next_token)) {
      break;
    }
    result.tokens.push_back(next_token);

    auto forward_or = runtime.ForwardToken(next_token, pos);
    if (!forward_or.ok()) {
      return forward_or.status();
    }
    next_token = forward_or->next_token;
  }

  result.text = tokenizer.Decode(result.tokens);
  LOG(INFO) << "Generated tokens: " << result.tokens.size();
  return result;
}

}  // namespace runtime
