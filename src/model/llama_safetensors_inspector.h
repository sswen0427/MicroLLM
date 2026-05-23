#pragma once

#include <string>

#include "absl/status/status.h"
#include "model/hf_config.h"

namespace model {

absl::Status InspectLlamaSafetensorsModel(const std::string& model_dir);

absl::Status InspectLlamaSafetensorsModel(const HfLlamaConfig& config,
                                          const std::string& safetensors_path);

}  // namespace model
