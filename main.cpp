#include <glog/logging.h>

#include "base/base.h"
#include "model/llama2.h"

int32_t generate(const model::LLama2Model& model, const std::string& sentence,
                 int total_steps, bool need_output = false) {
  // Step1: Encode the sentence to tokens.
  auto tokens = model.encode(sentence);
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  // Step2: Get the prompt embedding.
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor =
      model.get_buffer(model::ModelBufferType::kInputPos);

  int32_t prompt_len = tokens.size();
  int32_t pos = 0;
  int32_t next = -1;
  bool is_prompt = true;
  std::vector<int32_t> words;
  // Step3: Generate the sentence.
  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input =
          model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
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

    pos += 1;
  }
  if (need_output) {
    printf("%s ", model.decode(words).data());
    fflush(stdout);
  }
  return std::min(pos, total_steps);
}

// ./main /data00/home/wensisi.0427/MicroLLM/tools/chat_q8.bin
// /home/wensisi.0427/MicroLLM/tools/my_tinyllama/AI-ModelScope/TinyLlama-1.1B-Chat-v1.0/tokenizer.model
int main(int argc, char* argv[]) {
  CHECK_EQ(argc, 3) << "Usage: ./main checkpoint_path tokenizer_path";
  const std::string& checkpoint_path = argv[1];  // e.g. out/model.bin
  const std::string& tokenizer_path = argv[2];

  // Step1: Init the model.
  model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path,
                           checkpoint_path, false);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: "
               << init_status.get_err_code();
  }

  // Step2: Generate the sentence.
  LOG(INFO) << "Start Generating...";
  const auto start = std::chrono::steady_clock::now();
  const std::string& sentence = "hello";
  const int steps = generate(model, sentence, 128, true);
  const auto end = std::chrono::steady_clock::now();
  const auto duration = std::chrono::duration<double>(end - start).count();
  LOG(INFO) << "Finish Generating, the duration is: " << duration
            << ", the steps/s:" << steps / duration;
  return 0;
}
