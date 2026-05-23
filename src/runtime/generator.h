#pragma once

#include <cstdint>
#include <string>

#include "base/base.h"
#include "model/model.h"

namespace runtime {

struct GenerationConfig {
  int32_t max_steps = 128;
};

struct GenerationResult {
  base::Status status;
  std::string text;
  int32_t steps = 0;
};

GenerationResult Generate(const model::Model &model, const std::string &prompt,
                          const GenerationConfig &config);

}  // namespace runtime
