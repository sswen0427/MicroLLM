#ifndef MICROLLM_INCLUDE_MODEL_MODEL_H
#define MICROLLM_INCLUDE_MODEL_MODEL_H
#include <string>
#include <vector>

#include "base/base.h"
#include "config.h"
#include "encode.h"
#include "op/embedding.h"
#include "tensor/tensor.h"

namespace model {
enum class ModelBufferType {
  kInputTokens = 0,
  kInputEmbeddings = 1,
  kOutputRMSNorm = 2,
  kKeyCache = 3,
  kValueCache = 4,
  kQuery = 5,
  kInputPos = 6,
  kScoreStorage = 7,
  kOutputMHA = 8,
  kAttnOutput = 9,
  kW1Output = 10,
  kW2Output = 11,
  kW3Output = 12,
  kFFNRMSNorm = 13,
  kForwardOutput = 15,
  kForwardOutputCPU = 16,

  kSinCache = 17,
  kCosCache = 18,
};

class Model {
 public:
  Model(const base::TokenizerType& tokenizer_type,
        const base::ModelType& model_type, std::string token_path,
        std::string model_path, bool is_quant_model);

  virtual base::Status init(base::DeviceType device_type) = 0;

  virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx) = 0;

  virtual const tensor::Tensor& get_buffer(
      ModelBufferType buffer_idx) const = 0;

  virtual bool is_sentence_ending(int32_t token_idx) const = 0;

  virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;

  virtual std::string decode(int32_t token_idx) const = 0;

  virtual std::string decode(std::vector<int32_t> token_idxs) const = 0;

  virtual op::EmbeddingOutput embedding(
      const std::vector<int>& tokens) const = 0;

  virtual base::Status predict(const tensor::Tensor& input,
                               const tensor::Tensor& pos_tensor, bool is_prompt,
                               int& next) const = 0;

  virtual tensor::Tensor fill_input(const tensor::Tensor& pos_tensor,
                                    const op::EmbeddingOutput& embedding_output,
                                    bool is_prompt) const = 0;

 protected:
  virtual base::Status gen_model_from_file();

  virtual base::Status create_encode_layer();

 protected:
  const base::TokenizerType tokenizer_type_ =
      base::TokenizerType::kEncodeUnknown;
  const base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;
  const std::string token_path_;
  const std::string model_path_;
  const bool is_quant_model_ = false;
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
  std::unique_ptr<TransformerConfig> config_;
  std::unique_ptr<op::EncodeLayerBase> encode_layer_;
};
}  // namespace model

#endif  // MICROLLM_INCLUDE_MODEL_MODEL_H
