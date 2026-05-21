#include "cli/cli_options.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <iostream>
#include <optional>
#include <string_view>

namespace cli {
namespace {

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

bool ParseBool(std::string_view value, bool *out) {
  const std::string lower = ToLower(std::string(value));
  if (lower == "1" || lower == "true" || lower == "yes" || lower == "on") {
    *out = true;
    return true;
  }
  if (lower == "0" || lower == "false" || lower == "no" || lower == "off") {
    *out = false;
    return true;
  }
  return false;
}

bool SetOption(std::string_view key, std::optional<std::string_view> value,
               CliOptions *options, std::string *error) {
  auto require_value = [&]() -> std::optional<std::string_view> {
    if (!value || value->empty()) {
      *error = "Missing value for --" + std::string(key);
      return std::nullopt;
    }
    return value;
  };

  if (key == "help" || key == "h") {
    options->help = true;
    return true;
  }
  if (key == "quantized") {
    if (!value) {
      options->quantized = true;
      return true;
    }
    if (!ParseBool(*value, &options->quantized)) {
      *error = "Invalid boolean for --quantized: " + std::string(*value);
      return false;
    }
    return true;
  }

  const auto text = require_value();
  if (!text) {
    return false;
  }

  if (key == "model-type") {
    options->model_type = ToLower(std::string(*text));
  } else if (key == "checkpoint") {
    options->checkpoint_path = std::string(*text);
  } else if (key == "tokenizer") {
    options->tokenizer_path = std::string(*text);
  } else if (key == "tokenizer-type") {
    options->tokenizer_type = ToLower(std::string(*text));
  } else if (key == "prompt") {
    options->prompt = std::string(*text);
  } else if (key == "steps") {
    try {
      options->steps = std::stoi(std::string(*text));
    } catch (const std::exception &) {
      *error = "Invalid integer for --steps: " + std::string(*text);
      return false;
    }
  } else if (key == "device") {
    options->device = ToLower(std::string(*text));
  } else {
    *error = "Unknown option --" + std::string(key);
    return false;
  }
  return true;
}

} // namespace

void PrintUsage(std::ostream &os, const char *program) {
  os << "Usage:\n"
     << "  " << program
     << " --checkpoint <path> --tokenizer <path> [options]\n\n"
     << "Options:\n"
     << "  --model-type <llama2>      Model family. qwen2/qwen3 are not wired "
        "into the CLI yet.\n"
     << "  --checkpoint <path>        MicroLLM checkpoint file.\n"
     << "  --tokenizer <path>         Tokenizer model path.\n"
     << "  --tokenizer-type <spe>     Tokenizer type. Currently spe is the "
        "stable path.\n"
     << "  --prompt <text>            Prompt text. Default: hello\n"
     << "  --steps <n>                Maximum generation steps. Default: 128\n"
     << "  --device <cpu|cuda>        Runtime device. Default: cuda\n"
     << "  --quantized                Load checkpoint as int8 Q8_0 weights.\n"
     << "  --help                     Show this message.\n\n"
     << "Legacy form is also supported:\n"
     << "  " << program << " <checkpoint_path> <tokenizer_path>\n";
}

bool ParseCliOptions(int argc, char *argv[], CliOptions *options,
                     std::string *error) {
  if (argc == 3 && std::string_view(argv[1]).rfind("--", 0) != 0) {
    options->checkpoint_path = argv[1];
    options->tokenizer_path = argv[2];
    return true;
  }

  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg.rfind("--", 0) != 0) {
      *error = "Unexpected positional argument: " + std::string(arg);
      return false;
    }

    arg.remove_prefix(2);
    std::string_view key = arg;
    std::optional<std::string_view> value;
    const size_t eq_pos = arg.find('=');
    if (eq_pos != std::string_view::npos) {
      key = arg.substr(0, eq_pos);
      value = arg.substr(eq_pos + 1);
    } else if (key != "help" && key != "h" && key != "quantized") {
      if (i + 1 >= argc) {
        *error = "Missing value for --" + std::string(key);
        return false;
      }
      value = std::string_view(argv[++i]);
    }

    if (!SetOption(key, value, options, error)) {
      return false;
    }
  }

  return true;
}

bool ValidateCliOptions(const CliOptions &options, std::string *error) {
  if (options.help) {
    return true;
  }
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
