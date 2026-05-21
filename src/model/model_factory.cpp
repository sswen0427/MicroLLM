#include "model/model_factory.h"

#include <absl/status/status.h>

#include "model/llama2.h"

namespace model {

absl::StatusOr<std::unique_ptr<Model>>
CreateModel(const ModelFactoryConfig &config) {
  if (config.tokenizer_type == base::TokenizerType::kEncodeUnknown) {
    return absl::InvalidArgumentError("Unsupported tokenizer type.");
  }

  if (config.model_type == "llama2" || config.model_type == "llama" ||
      config.model_type == "tinyllama") {
    return std::make_unique<LLama2Model>(
        config.tokenizer_type, config.tokenizer_path, config.checkpoint_path,
        config.quantized);
  }

  if (config.model_type == "qwen2" || config.model_type == "qwen3") {
    return absl::InvalidArgumentError(
        config.model_type +
        " is present in the source tree but is not wired into the current "
        "CLI/build path yet.");
  }

  return absl::InvalidArgumentError("Unsupported model type: " +
                                    config.model_type);
}

} // namespace model
