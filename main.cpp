#include <glog/logging.h>

#include <chrono>
#include <iostream>
#include <string>

#include "base/base.h"
#include "cli/cli_options.h"
#include "model/model.h"
#include "model/model_factory.h"
#include "runtime/generator.h"

namespace {

model::ModelFactoryConfig BuildModelConfig(const cli::CliOptions &options) {
  model::ModelFactoryConfig config;
  config.model_type = options.model_type;
  config.tokenizer_type = cli::ParseTokenizerType(options.tokenizer_type);
  config.tokenizer_path = options.tokenizer_path;
  config.checkpoint_path = options.checkpoint_path;
  config.quantized = options.quantized;
  return config;
}

} // namespace

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  auto options_or = cli::ParseCliOptions(argc, argv);
  if (!options_or.ok()) {
    std::cerr << "Error: " << options_or.status().message()
              << "\nUse --help to see available flags.\n";
    return 1;
  }
  const cli::CliOptions &options = *options_or;

  model::ModelFactoryConfig model_config = BuildModelConfig(options);
  if (model_config.tokenizer_type == base::TokenizerType::kEncodeUnknown) {
    std::cerr << "Error: Unsupported tokenizer type: " << options.tokenizer_type
              << "\n";
    return 1;
  }

  auto model_or = model::CreateModel(model_config);
  if (!model_or.ok()) {
    std::cerr << "Error: " << model_or.status().message() << "\n";
    return 1;
  }
  auto model = std::move(*model_or);

  const base::DeviceType device_type = cli::ParseDevice(options.device);
  const auto init_status = model->init(device_type);
  if (!init_status) {
    LOG(FATAL) << "Model init failed, error code: "
               << init_status.get_err_code()
               << ", message: " << init_status.get_err_msg();
  }

  LOG(INFO) << "Start generating with model_type=" << options.model_type
            << ", device=" << options.device
            << ", quantized=" << options.quantized
            << ", steps=" << options.steps;

  const auto start = std::chrono::steady_clock::now();
  runtime::GenerationConfig generation_config;
  generation_config.max_steps = options.steps;
  const runtime::GenerationResult generation_result =
      runtime::Generate(*model, options.prompt, generation_config);
  const auto end = std::chrono::steady_clock::now();
  if (!generation_result.status) {
    std::cerr << "Error: " << generation_result.status.get_err_msg() << "\n";
    return 1;
  }

  std::cout << generation_result.text << std::endl;
  const double duration = std::chrono::duration<double>(end - start).count();

  LOG(INFO) << "Finish generating, duration: " << duration
            << "s, steps/s: " << generation_result.steps / duration;
  return 0;
}
