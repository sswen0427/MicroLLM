#pragma once

#include <absl/status/statusor.h>

#include <string>

#include "base/base.h"

namespace cli {

struct CliOptions {
  std::string model_type = "llama2";
  std::string checkpoint_path;
  std::string tokenizer_path;
  std::string tokenizer_type = "spe";
  std::string prompt = "hello";
  std::string device = "cuda";
  int32_t steps = 128;
  bool quantized = false;
};

absl::StatusOr<CliOptions> ParseCliOptions(int argc, char *argv[]);

base::DeviceType ParseDevice(const std::string &device);

base::TokenizerType ParseTokenizerType(const std::string &tokenizer_type);

}  // namespace cli
