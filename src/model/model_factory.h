#pragma once

#include <absl/status/statusor.h>

#include <memory>
#include <string>

#include "base/types.h"
#include "model/model.h"

namespace model {

struct ModelFactoryConfig {
  std::string model_type = "llama2";
  base::TokenizerType tokenizer_type = base::TokenizerType::kEncodeSpe;
  std::string tokenizer_path;
  std::string checkpoint_path;
  bool quantized = false;
};

absl::StatusOr<std::unique_ptr<Model>> CreateModel(
    const ModelFactoryConfig &config);

}  // namespace model
