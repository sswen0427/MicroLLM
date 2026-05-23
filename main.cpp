#include <glog/logging.h>

#include <absl/status/status.h>
#include <gflags/gflags.h>

#include <iostream>
#include <string>

#include "model/llama_safetensors_inspector.h"

DEFINE_string(model_dir, "", "HuggingFace model directory.");

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

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

  const absl::Status status =
      model::InspectLlamaSafetensorsModel(FLAGS_model_dir, std::cout);
  if (!status.ok()) {
    std::cerr << "Error: " << status.message() << "\n";
    return 1;
  }

  LOG(INFO) << "Model directory inspection finished: " << FLAGS_model_dir;
  return 0;
}
