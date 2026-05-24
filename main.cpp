#include <absl/status/status.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <filesystem>
#include <iostream>

#include "model/llama_hf_model_loader.h"

DEFINE_string(model_dir, "", "HuggingFace model directory.");

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "MicroLLM inference runtime.\n\n"
      "Usage:\n"
      "  MicroLLM --model_dir <hf_model_dir>");

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

  LOG(INFO) << "Model directory loading finished: " << FLAGS_model_dir;
  return 0;
}
