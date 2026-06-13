#include "runtime/generator.h"

#include <absl/status/status.h>
#include <absl/strings/str_cat.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "base/types.h"
#include "tensor/tensor.h"

namespace runtime {
namespace {

std::vector<std::pair<int32_t, float>> TopLogits(const tensor::Tensor& logits,
                                                 int32_t top_k) {
  CHECK(logits.data_type() == base::DataType::kDataTypeFp32);
  CHECK(logits.device_type() == base::DeviceType::kDeviceCPU);
  const int32_t size = static_cast<int32_t>(logits.size());
  const int32_t k = std::min(top_k, size);
  std::vector<int32_t> indices(size);
  std::iota(indices.begin(), indices.end(), 0);
  const float* data = logits.data<float>();
  std::partial_sort(
      indices.begin(), indices.begin() + k, indices.end(),
      [data](int32_t lhs, int32_t rhs) { return data[lhs] > data[rhs]; });

  std::vector<std::pair<int32_t, float>> top_logits;
  top_logits.reserve(k);
  for (int32_t i = 0; i < k; ++i) {
    top_logits.emplace_back(indices[i], data[indices[i]]);
  }
  return top_logits;
}

}  // namespace

absl::StatusOr<GenerationResult> GenerateText(
    const model::LlamaHfModel& model, const tokenizer::Tokenizer& tokenizer,
    const std::string& prompt, const GenerationConfig& config) {
  if (prompt.empty()) {
    return absl::InvalidArgumentError("prompt must not be empty.");
  }
  if (config.max_new_tokens <= 0) {
    return absl::InvalidArgumentError("max_new_tokens must be greater than 0.");
  }
  if (config.trace_steps && config.trace_top_k <= 0) {
    return absl::InvalidArgumentError("trace_top_k must be greater than 0.");
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
  GenerationResult result;
  result.prompt_tokens = prompt_tokens;
  result.tokens.reserve(config.max_new_tokens);
  result.profile.prompt_tokens = prompt_tokens.size();
  if (config.trace_steps) {
    result.steps.reserve(config.max_new_tokens);
  }

  int32_t next_token = -1;
  model::LlamaForwardResult last_forward;
  for (int32_t pos = 0; pos < static_cast<int32_t>(prompt_tokens.size());
       ++pos) {
    auto forward_or = runtime.ForwardToken(prompt_tokens[pos], pos);
    if (!forward_or.ok()) {
      return forward_or.status();
    }
    last_forward = std::move(*forward_or);
    next_token = last_forward.next_token;
  }

  int32_t pos = static_cast<int32_t>(prompt_tokens.size());
  for (int32_t step = 0; step < config.max_new_tokens; ++step, ++pos) {
    if (next_token < 0 || tokenizer.IsEndToken(next_token)) {
      break;
    }
    if (config.trace_steps) {
      result.steps.push_back(GenerationStep{
          .step = step,
          .position = pos - 1,
          .input_token_id =
              step == 0 ? prompt_tokens.back() : result.tokens.back(),
          .next_token_id = next_token,
          .top_logits = TopLogits(last_forward.logits, config.trace_top_k),
      });
    }
    result.tokens.push_back(next_token);

    auto forward_or = runtime.ForwardToken(next_token, pos);
    if (!forward_or.ok()) {
      return forward_or.status();
    }
    last_forward = std::move(*forward_or);
    next_token = last_forward.next_token;
  }

  result.text = tokenizer.Decode(result.tokens);
  result.profile.generated_tokens = result.tokens.size();
  result.profile.forward = runtime.profile();
  LOG(INFO) << "Generated tokens: " << result.tokens.size();
  return result;
}

void GenerationProfile::Log() const {
  LOG(INFO) << "=== Generation Profile ===";
  LOG(INFO) << "prompt_tokens=" << prompt_tokens;
  LOG(INFO) << "generated_tokens=" << generated_tokens;
  forward.Log();
}

}  // namespace runtime
