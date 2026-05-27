#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>

#include "model/llama_hf_model_loader.h"
#include "runtime/generator.h"
#include "tokenizer/tokenizer.h"

DEFINE_string(model_dir, "", "HuggingFace model directory.");
DEFINE_string(prompt, "", "Prompt text to generate from.");
DEFINE_int32(max_new_tokens, 32, "Maximum number of tokens to generate.");

namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void PrintProfile(double model_load_ms, double tokenizer_load_ms,
                  double generation_ms,
                  const runtime::GenerationProfile& profile) {
  const double decode_ms_per_token =
      profile.generated_tokens == 0
          ? 0.0
          : profile.decode_ms / static_cast<double>(profile.generated_tokens);
  const double tokens_per_second =
      profile.decode_ms <= 0.0
          ? 0.0
          : static_cast<double>(profile.generated_tokens) * 1000.0 /
                profile.decode_ms;

  std::cerr << std::fixed << std::setprecision(3)
            << "Profile:\n"
            << "  model_load_ms: " << model_load_ms << "\n"
            << "  tokenizer_load_ms: " << tokenizer_load_ms << "\n"
            << "  generation_ms: " << generation_ms << "\n"
            << "  prompt_tokens: " << profile.prompt_tokens << "\n"
            << "  generated_tokens: " << profile.generated_tokens << "\n"
            << "  prefill_ms: " << profile.prefill_ms << "\n"
            << "  decode_ms: " << profile.decode_ms << "\n"
            << "  decode_ms_per_token: " << decode_ms_per_token << "\n"
            << "  decode_tokens_per_second: " << tokens_per_second << "\n";
}

}  // namespace

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

  const Clock::time_point model_load_start = Clock::now();
  auto model_or = model::LoadLlamaHfModel(FLAGS_model_dir);
  const Clock::time_point model_load_end = Clock::now();
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

  const Clock::time_point tokenizer_load_start = Clock::now();
  auto tokenizer_or = tokenizer::SentencePieceTokenizer::Load(
      tokenizer_path.string(),
      tokenizer::TokenizerOptions{.add_bos = true, .add_eos = false});
  const Clock::time_point tokenizer_load_end = Clock::now();
  if (!tokenizer_or.ok()) {
    std::cerr << "Error: " << tokenizer_or.status().message() << "\n";
    return 1;
  }

  runtime::GenerationConfig generation_config;
  generation_config.max_new_tokens = FLAGS_max_new_tokens;
  const Clock::time_point generation_start = Clock::now();
  auto result_or = runtime::GenerateText(**model_or, **tokenizer_or,
                                         FLAGS_prompt, generation_config);
  const Clock::time_point generation_end = Clock::now();
  if (!result_or.ok()) {
    std::cerr << "Error: " << result_or.status().message() << "\n";
    return 1;
  }

  PrintProfile(ElapsedMs(model_load_start, model_load_end),
               ElapsedMs(tokenizer_load_start, tokenizer_load_end),
               ElapsedMs(generation_start, generation_end),
               result_or->profile);
  std::cout << result_or->text << "\n";
  return 0;
}
