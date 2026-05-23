#include "cli/cli_options.h"

#include <absl/status/status.h>
#include <absl/strings/ascii.h>
#include <gflags/gflags.h>

DEFINE_string(model_type, "llama2",
              "Model family. Supported stable path: llama2.");
DEFINE_string(model_dir, "", "HuggingFace model directory.");
DEFINE_string(checkpoint, "", "MicroLLM checkpoint file.");
DEFINE_string(tokenizer, "", "Tokenizer model path.");
DEFINE_string(tokenizer_type, "spe",
              "Tokenizer type. Currently spe is the stable path.");
DEFINE_string(prompt, "hello", "Prompt text.");
DEFINE_string(device, "cuda", "Runtime device: cpu or cuda.");
DEFINE_int32(steps, 128, "Maximum generation steps.");
DEFINE_bool(quantized, false, "Load checkpoint as int8 Q8_0 weights.");
DEFINE_bool(inspect_model, false,
            "Inspect a HuggingFace safetensors model directory and exit.");

namespace cli {
namespace {

absl::Status ValidateCliOptions(const CliOptions &options) {
  if (options.inspect_model) {
    if (options.model_dir.empty()) {
      return absl::InvalidArgumentError(
          "--model_dir is required when --inspect_model is set.");
    }
    return absl::OkStatus();
  }
  if (options.checkpoint_path.empty()) {
    return absl::InvalidArgumentError("--checkpoint is required.");
  }
  if (options.tokenizer_path.empty()) {
    return absl::InvalidArgumentError("--tokenizer is required.");
  }
  if (options.steps <= 0) {
    return absl::InvalidArgumentError("--steps must be greater than 0.");
  }
  if (ParseDevice(options.device) == base::DeviceType::kDeviceUnknown) {
    return absl::InvalidArgumentError("Unsupported device: " + options.device);
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<CliOptions> ParseCliOptions(int argc, char *argv[]) {
  gflags::SetUsageMessage(
      "MicroLLM inference runtime.\n\n"
      "Usage:\n"
      "  MicroLLM --checkpoint <path> --tokenizer <path> [options]\n"
      "  MicroLLM --model_dir <hf_model_dir> --inspect_model");

  int parsed_argc = argc;
  char **parsed_argv = argv;
  gflags::ParseCommandLineFlags(&parsed_argc, &parsed_argv, true);

  if (parsed_argc > 1) {
    return absl::InvalidArgumentError("Unexpected positional argument: " +
                                      std::string(parsed_argv[1]));
  }

  CliOptions options;
  options.model_type = absl::AsciiStrToLower(FLAGS_model_type);
  options.model_dir = FLAGS_model_dir;
  options.checkpoint_path = FLAGS_checkpoint;
  options.tokenizer_path = FLAGS_tokenizer;
  options.tokenizer_type = absl::AsciiStrToLower(FLAGS_tokenizer_type);
  options.prompt = FLAGS_prompt;
  options.device = absl::AsciiStrToLower(FLAGS_device);
  options.steps = FLAGS_steps;
  options.quantized = FLAGS_quantized;
  options.inspect_model = FLAGS_inspect_model;

  const absl::Status status = ValidateCliOptions(options);
  if (!status.ok()) {
    return status;
  }
  return options;
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

}  // namespace cli
