#include "runtime/generator.h"

#include <algorithm>
#include <vector>

#include <glog/logging.h>

namespace runtime {

GenerationResult Generate(const model::Model &model, const std::string &prompt,
                          const GenerationConfig &config) {
  if (config.max_steps <= 0) {
    return {base::error::InvalidArgument("max_steps must be greater than 0."),
            "", 0};
  }

  auto tokens = model.encode(prompt);
  if (tokens.empty()) {
    return {base::error::InvalidArgument("The token list is empty."), "", 0};
  }

  const auto &prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor =
      model.get_buffer(model::ModelBufferType::kInputPos);

  const int32_t prompt_len = static_cast<int32_t>(tokens.size());
  int32_t pos = 0;
  int32_t next = -1;
  bool is_prompt = true;
  std::vector<int32_t> words;

  while (pos < config.max_steps) {
    pos_tensor.at<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input =
          model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      const base::Status status =
          model.predict(input, pos_tensor, is_prompt, next);
      if (!status) {
        return {status, "", pos};
      }
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto &token_embedding = model.embedding(tokens);
      tensor::Tensor input =
          model.fill_input(pos_tensor, token_embedding, is_prompt);
      const base::Status status =
          model.predict(input, pos_tensor, is_prompt, next);
      if (!status) {
        return {status, "", pos};
      }
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

  return {base::error::Success(), model.decode(words),
          std::min(pos, config.max_steps)};
}

} // namespace runtime
