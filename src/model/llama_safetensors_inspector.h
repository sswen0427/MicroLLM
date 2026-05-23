#pragma once

#include <absl/status/status.h>

#include <string>

namespace model {

absl::Status InspectLlamaSafetensorsModel(const std::string& model_dir);

}  // namespace model
