#include "cli/cli_options.h"

#include <algorithm>
#include <cctype>
#include <string_view>

#include <gflags/gflags.h>

DEFINE_string(model_type, "llama2",
              "Model family. Supported stable path: llama2.");
DEFINE_string(checkpoint, "", "MicroLLM checkpoint file.");
DEFINE_string(tokenizer, "", "Tokenizer model path.");
DEFINE_string(tokenizer_type, "spe",
              "Tokenizer type. Currently spe is the stable path.");
DEFINE_string(prompt, "hello", "Prompt text.");
DEFINE_string(device, "cuda", "Runtime device: cpu or cuda.");
DEFINE_int32(steps, 128, "Maximum generation steps.");
DEFINE_bool(quantized, false, "Load checkpoint as int8 Q8_0 weights.");

namespace cli {
namespace {

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

} // namespace

bool ParseCliOptions(int argc, char *argv[], CliOptions *options,
                     std::string *error) {
  gflags::SetUsageMessage(
      "MicroLLM inference runtime.\n\n"
      "Usage:\n"
      "  MicroLLM --checkpoint <path> --tokenizer <path> [options]\n"
      "  MicroLLM <checkpoint_path> <tokenizer_path>");

  int parsed_argc = argc;
  char **parsed_argv = argv;
  gflags::ParseCommandLineFlags(&parsed_argc, &parsed_argv, true);

  if (parsed_argc == 3 &&
      std::string_view(parsed_argv[1]).rfind("--", 0) != 0) {
    FLAGS_checkpoint = parsed_argv[1];
    FLAGS_tokenizer = parsed_argv[2];
  } else if (parsed_argc > 1) {
    *error = "Unexpected positional argument: " + std::string(parsed_argv[1]);
    return false;
  }

  options->model_type = ToLower(FLAGS_model_type);
  options->checkpoint_path = FLAGS_checkpoint;
  options->tokenizer_path = FLAGS_tokenizer;
  options->tokenizer_type = ToLower(FLAGS_tokenizer_type);
  options->prompt = FLAGS_prompt;
  options->device = ToLower(FLAGS_device);
  options->steps = FLAGS_steps;
  options->quantized = FLAGS_quantized;
  return true;
}

bool ValidateCliOptions(const CliOptions &options, std::string *error) {
  if (options.checkpoint_path.empty()) {
    *error = "--checkpoint is required.";
    return false;
  }
  if (options.tokenizer_path.empty()) {
    *error = "--tokenizer is required.";
    return false;
  }
  if (options.steps <= 0) {
    *error = "--steps must be greater than 0.";
    return false;
  }
  if (ParseDevice(options.device) == base::DeviceType::kDeviceUnknown) {
    *error = "Unsupported device: " + options.device;
    return false;
  }
  return true;
}

base::DeviceType ParseDevice(const std::string &device) {
  if (device == "cpu") {
    return base::DeviceType::kDeviceCPU;
  }
  if (device == "cuda") {
    return base::DeviceType::kDeviceCUDA;
  }
  return base::DeviceType::kDeviceUnknown;
}

base::TokenizerType ParseTokenizerType(const std::string &tokenizer_type) {
  if (tokenizer_type == "spe" || tokenizer_type == "sentencepiece") {
    return base::TokenizerType::kEncodeSpe;
  }
  return base::TokenizerType::kEncodeUnknown;
}

} // namespace cli
