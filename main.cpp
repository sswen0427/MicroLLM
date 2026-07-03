#include <gflags/gflags.h>
#include <glog/logging.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "base/types.h"
#include "model/llama_hf_model_loader.h"
#include "runtime/chat_template.h"
#include "runtime/generator.h"
#include "tokenizer/tokenizer.h"

DEFINE_string(model_dir, "", "HuggingFace model directory.");
DEFINE_int32(max_new_tokens, 32, "Maximum number of tokens to generate.");
DEFINE_string(device, "cpu", "Inference device: cpu or cuda.");

namespace {

base::DeviceType ParseDeviceType(const std::string& device) {
  if (device == "cpu") {
    return base::DeviceType::kDeviceCPU;
  }
  if (device == "cuda") {
    return base::DeviceType::kDeviceCUDA;
  }
  return base::DeviceType::kDeviceUnknown;
}

}  // namespace

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "MicroLLM inference runtime.\n\n"
      "Usage:\n"
      "  MicroLLM --model_dir <hf_model_dir> [--device cpu|cuda]");

  int parsed_argc = argc;
  char** parsed_argv = argv;
  gflags::ParseCommandLineFlags(&parsed_argc, &parsed_argv, true);

  if (parsed_argc > 1) {
    std::cerr << "Error: Unexpected positional argument: " << parsed_argv[1]
              << "\nUse --help to see available flags.\n";
    return 1;
  }
  if (FLAGS_model_dir.empty()) {
    std::cerr << "Error: --model_dir is required.\n"
              << "Use --help to see available flags.\n";
    return 1;
  }
  if (FLAGS_max_new_tokens <= 0) {
    std::cerr << "Error: --max_new_tokens must be greater than 0.\n";
    return 1;
  }
  const base::DeviceType device_type = ParseDeviceType(FLAGS_device);
  if (device_type == base::DeviceType::kDeviceUnknown) {
    std::cerr << "Error: --device must be cpu or cuda.\n";
    return 1;
  }
  const std::filesystem::path log_dir =
      std::filesystem::absolute(std::filesystem::path(FLAGS_model_dir)) /
      "logs";
  std::filesystem::create_directories(log_dir);
  FLAGS_log_dir = log_dir.string();
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  LOG(INFO) << "Writing logs to model directory: " << FLAGS_log_dir;

  auto model_or = model::LoadLlamaHfModel(FLAGS_model_dir);
  if (!model_or.ok()) {
    std::cerr << "Error: " << model_or.status().message() << "\n";
    return 1;
  }

  const std::filesystem::path tokenizer_path =
      std::filesystem::path(FLAGS_model_dir) / "tokenizer.model";
  if (!std::filesystem::exists(tokenizer_path)) {
    std::cerr << "Error: tokenizer.model does not exist: "
              << tokenizer_path.string() << "\n";
    return 1;
  }

  auto tokenizer_or = tokenizer::SentencePieceTokenizer::Load(
      tokenizer_path.string(),
      tokenizer::TokenizerOptions{.add_bos = true, .add_eos = false});
  if (!tokenizer_or.ok()) {
    std::cerr << "Error: " << tokenizer_or.status().message() << "\n";
    return 1;
  }

  runtime::GenerationConfig generation_config;
  generation_config.max_new_tokens = FLAGS_max_new_tokens;
  generation_config.device_type = device_type;

  std::vector<runtime::ChatMessage> messages;

  std::string pending_user_message;
  while (true) {
    if (pending_user_message.empty()) {
      std::cout << "User> ";
      if (!std::getline(std::cin, pending_user_message)) {
        break;
      }
    }
    if (pending_user_message == "exit" || pending_user_message == "quit") {
      break;
    }
    if (pending_user_message.empty()) {
      continue;
    }

    messages.push_back(
        {.role = runtime::ChatRole::kUser, .content = pending_user_message});
    const std::string prompt = runtime::FormatTinyLlamaChatPrompt(messages);
    auto result_or = runtime::GenerateText(**model_or, **tokenizer_or, prompt,
                                           generation_config);
    if (!result_or.ok()) {
      std::cerr << "Error: " << result_or.status().message() << "\n";
      return 1;
    }

    result_or->profile.Log();
    std::cout << "Assistant> " << result_or->text << "\n";
    messages.push_back(
        {.role = runtime::ChatRole::kAssistant, .content = result_or->text});
    pending_user_message.clear();
  }
  return 0;
}
