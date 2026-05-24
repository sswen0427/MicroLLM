#pragma once

#include <absl/status/status.h>

#include <map>
#include <string>
#include <vector>

#include "base/types.h"
#include "model/config.h"
#include "model/raw_model_data.h"
#include "op/embedding.h"
#include "op/encode.h"
#include "sampler/sampler.h"
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
  explicit Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
                 std::string token_path, std::string model_path,
                 bool is_quant_model);

  virtual absl::Status init(base::DeviceType device_type) = 0;

  virtual absl::Status predict(const tensor::Tensor& input,
                               const tensor::Tensor& pos_tensor, bool is_prompt,
                               int& next) const = 0;

  virtual absl::Status forward(const tensor::Tensor& input,
                               const tensor::Tensor& pos_tensor,
                               int& next) const = 0;

  base::ModelType model_type() const;

  const std::string& token_path() const;

  const std::string& model_path() const;

  virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx);

  virtual const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const;

  virtual bool is_sentence_ending(int32_t token_idx) const;

  virtual std::string decode(int32_t token_idx) const;

  virtual std::string decode(std::vector<int32_t> token_idxs) const;

  /////////////////////////////////////////////////////
  /////////////////////////////////////////////////////
  virtual std::vector<int32_t> encode(const std::string& sentence) const;

  virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(
      int32_t layer_idx, int32_t token_pos) const;

  virtual op::EmbeddingOutput embedding(
      const std::vector<int>& tokens) const = 0;

  virtual tensor::Tensor fill_input(const tensor::Tensor& pos_tensor,
                                    const op::EmbeddingOutput& embedding_output,
                                    bool is_prompt) const;

 protected:
  virtual absl::Status insert_buffer(ModelBufferType buffer_idx,
                                     const tensor::Tensor& tensor);

  virtual absl::Status read_model_file();

  virtual absl::Status create_encode_layer();

  virtual absl::Status gen_model_from_file();

  virtual absl::Status generate_model_infos(const ModelConfig& config) const;

  virtual int32_t post_processing(const tensor::Tensor& pos,
                                  bool is_prompt) const = 0;

 private:
  virtual void init_mem() = 0;

  virtual absl::Status create_layers() = 0;

  virtual void create_param_layers() = 0;

  virtual void create_nonparam_layers() = 0;

  virtual void create_param_quant_layers() = 0;

 protected:
  int32_t group_size_ = 1;
  bool is_quant_model_ = false;
  std::unique_ptr<TransformerConfig> config_;

  std::string token_path_;
  std::string model_path_;
  std::unique_ptr<op::EncodeLayerBase> encode_layer_;
  std::map<ModelBufferType, tensor::Tensor> buffers_;
  std::unique_ptr<sampler::Sampler> sampler_;
  std::shared_ptr<RawModelData> raw_model_data_;
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
  base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;
  base::TokenizerType tokenizer_type_ = base::TokenizerType::kEncodeUnknown;
};
}  // namespace model
