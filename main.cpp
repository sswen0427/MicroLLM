#include <gflags/gflags.h>
#include <glog/logging.h>

#include <filesystem>
#include <iostream>

#include "model/llama_hf_model_loader.h"
#include "runtime/generator.h"
#include "tokenizer/tokenizer.h"

DEFINE_string(model_dir, "", "HuggingFace model directory.");
DEFINE_string(prompt, "", "Prompt text to generate from.");
DEFINE_int32(max_new_tokens, 32, "Maximum number of tokens to generate.");

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "MicroLLM inference runtime.\n\n"
      "Usage:\n"
      "  MicroLLM --model_dir <hf_model_dir> --prompt <text>");

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
  if (FLAGS_prompt.empty()) {
    std::cerr << "Error: --prompt is required.\n"
              << "Use --help to see available flags.\n";
    return 1;
  }
  if (FLAGS_max_new_tokens <= 0) {
    std::cerr << "Error: --max_new_tokens must be greater than 0.\n";
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
  auto result_or = runtime::GenerateText(**model_or, **tokenizer_or,
                                         FLAGS_prompt, generation_config);
  if (!result_or.ok()) {
    std::cerr << "Error: " << result_or.status().message() << "\n";
    return 1;
  }

  result_or->profile.Log();
  std::cout << result_or->text << "\n";
  return 0;
}
