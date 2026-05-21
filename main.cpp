#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "base/base.h"
#include "cli/cli_options.h"
#include "model/model.h"
#include "model/model_factory.h"

namespace {

int32_t Generate(const model::Model &model, const std::string &prompt,
                 int32_t total_steps, bool need_output = false) {
  auto tokens = model.encode(prompt);
  LOG_IF(FATAL, tokens.empty()) << "The token list is empty.";

  const auto &prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor =
      model.get_buffer(model::ModelBufferType::kInputPos);

  const int32_t prompt_len = static_cast<int32_t>(tokens.size());
  int32_t pos = 0;
  int32_t next = -1;
  bool is_prompt = true;
  std::vector<int32_t> words;

  while (pos < total_steps) {
    pos_tensor.at<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input =
          model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto &token_embedding = model.embedding(tokens);
      tensor::Tensor input =
          model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    }

    if (model.is_sentence_ending(next)) {
      break;
    }
    if (is_prompt) {
      next = tokens.at(pos + 1);
      words.push_back(next);
    } else {
      words.push_back(next);
    }
    ++pos;
  }

  if (need_output) {
    std::cout << model.decode(words) << std::flush;
  }
  return std::min(pos, total_steps);
}

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

  cli::CliOptions options;
  std::string error;
  if (!cli::ParseCliOptions(argc, argv, &options, &error) ||
      !cli::ValidateCliOptions(options, &error)) {
    std::cerr << "Error: " << error << "\nUse --help to see available flags.\n";
    return 1;
  }

  model::ModelFactoryConfig model_config = BuildModelConfig(options);
  if (model_config.tokenizer_type == base::TokenizerType::kEncodeUnknown) {
    std::cerr << "Error: Unsupported tokenizer type: " << options.tokenizer_type
              << "\n";
    return 1;
  }

  auto model = model::CreateModel(model_config, &error);
  if (!model) {
    std::cerr << "Error: " << error << "\n";
    return 1;
  }

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
  const int32_t steps = Generate(*model, options.prompt, options.steps, true);
  const auto end = std::chrono::steady_clock::now();
  const double duration = std::chrono::duration<double>(end - start).count();

  std::cout << std::endl;
  LOG(INFO) << "Finish generating, duration: " << duration
            << "s, steps/s: " << steps / duration;
  return 0;
}
