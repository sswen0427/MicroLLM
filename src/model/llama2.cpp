#include "model/llama2.h"

#include <iostream>
#include <string>
#include <vector>

namespace model {

LLama2Model::LLama2Model(base::TokenizerType tokenizer_type,
                         const std::string &token_path,
                         const std::string &model_path, bool is_quant_model)
    : Model(token_path, model_path) {
  std::cout << "[Stub] LLama2Model Constructor called. "
            << "token_path: " << token_path << ", model_path: " << model_path
            << std::endl;
}

base::Status LLama2Model::init(base::DeviceType device_type) { return {}; }

tensor::Tensor &LLama2Model::get_buffer(ModelBufferType buffer_idx) {
  static tensor::Tensor dummy_tensor;
  return dummy_tensor;
}

const tensor::Tensor &LLama2Model::get_buffer(
    ModelBufferType buffer_idx) const {
  static tensor::Tensor dummy_tensor;
  return dummy_tensor;
}

bool LLama2Model::is_sentence_ending(int32_t token_idx) const { return false; }

std::vector<int32_t> LLama2Model::encode(const std::string &sentence) const {
  return std::vector<int32_t>{};
}

std::string LLama2Model::decode(int32_t token_idx) const {
  return "dummy_word";
}

std::string LLama2Model::decode(std::vector<int32_t> token_idxs) const {
  return "dummy_sentence";
}

op::EmbeddingOutput LLama2Model::embedding(
    const std::vector<int> &tokens) const {
  return {};
}

base::Status LLama2Model::predict(const tensor::Tensor &input,
                                  const tensor::Tensor &pos_tensor,
                                  bool is_prompt, int &next) const {
  return {};
}

tensor::Tensor LLama2Model::fill_input(
    const tensor::Tensor &pos_tensor,
    const op::EmbeddingOutput &embedding_output, bool is_prompt) const {
  return tensor::Tensor();
}

}  // namespace model