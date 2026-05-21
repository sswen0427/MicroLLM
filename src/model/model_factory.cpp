#include "model/model_factory.h"

#include "model/llama2.h"

namespace model {

std::unique_ptr<Model> CreateModel(const ModelFactoryConfig &config,
                                   std::string *error) {
  if (config.tokenizer_type == base::TokenizerType::kEncodeUnknown) {
    if (error != nullptr) {
      *error = "Unsupported tokenizer type.";
    }
    return nullptr;
  }

  if (config.model_type == "llama2" || config.model_type == "llama" ||
      config.model_type == "tinyllama") {
    return std::make_unique<LLama2Model>(
        config.tokenizer_type, config.tokenizer_path, config.checkpoint_path,
        config.quantized);
  }

  if (config.model_type == "qwen2" || config.model_type == "qwen3") {
    if (error != nullptr) {
      *error = config.model_type +
               " is present in the source tree but is not wired into the "
               "current CLI/build path yet.";
    }
    return nullptr;
  }

  if (error != nullptr) {
    *error = "Unsupported model type: " + config.model_type;
  }
  return nullptr;
}

} // namespace model
