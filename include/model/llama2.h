#ifndef MICROLLM_INCLUDE_MODEL_LLAMA2_H
#define MICROLLM_INCLUDE_MODEL_LLAMA2_H

#include "base/base.h"
#include "model/model.h"

namespace model {
struct LLama2Layers {};

class LLama2Model : public Model {
 public:
  explicit LLama2Model(base::TokenizerType tokenizer_type,
                       const std::string &token_path,
                       const std::string &model_path, bool is_quant_model);

  base::Status init(base::DeviceType device_type) override;

  tensor::Tensor &get_buffer(ModelBufferType buffer_idx) override;

  const tensor::Tensor &get_buffer(ModelBufferType buffer_idx) const;

  bool is_sentence_ending(int32_t token_idx) const;

  std::vector<int32_t> encode(const std::string &sentence) const;

  std::string decode(int32_t token_idx) const;

  std::string decode(std::vector<int32_t> token_idxs) const;

  op::EmbeddingOutput embedding(const std::vector<int> &tokens) const override;

  base::Status predict(const tensor::Tensor &input,
                       const tensor::Tensor &pos_tensor, bool is_prompt,
                       int &next) const override;

  tensor::Tensor fill_input(const tensor::Tensor &pos_tensor,
                            const op::EmbeddingOutput &embedding_output,
                            bool is_prompt) const override;
};

}  // namespace model

#endif  // MICROLLM_INCLUDE_MODEL_LLAMA2_H
