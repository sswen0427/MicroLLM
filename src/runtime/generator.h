#pragma once

#include <absl/status/status.h>

#include <cstdint>
#include <string>

#include "base/types.h"
#include "model/model.h"

namespace runtime {

struct GenerationConfig {
  int32_t max_steps = 128;
};

struct GenerationResult {
  absl::Status status;
  std::string text;
  int32_t steps = 0;
};

GenerationResult Generate(const model::Model &model, const std::string &prompt,
                          const GenerationConfig &config);

}  // namespace runtime
