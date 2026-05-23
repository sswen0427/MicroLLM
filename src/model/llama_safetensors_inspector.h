#pragma once

#include <absl/status/status.h>

#include <string>

#include "model/hf_config.h"

namespace model {

absl::Status InspectLlamaSafetensorsModel(const std::string& model_dir);

absl::Status InspectLlamaSafetensorsModel(const HfLlamaConfig& config,
                                          const std::string& safetensors_path);

}  // namespace model
