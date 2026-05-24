#pragma once

#include <absl/status/statusor.h>

#include <memory>
#include <string>
#include <vector>

#include "model/hf_config.h"
#include "tensor/tensor.h"

namespace model {

struct LlamaHfLayerWeights {
  tensor::Tensor input_layernorm;
  tensor::Tensor post_attention_layernorm;
  tensor::Tensor q_proj;
  tensor::Tensor k_proj;
  tensor::Tensor v_proj;
  tensor::Tensor o_proj;
  tensor::Tensor gate_proj;
  tensor::Tensor up_proj;
  tensor::Tensor down_proj;
};

struct LlamaHfModelWeights {
  tensor::Tensor token_embedding;
  tensor::Tensor final_norm;
  tensor::Tensor lm_head;
  std::vector<LlamaHfLayerWeights> layers;
};

struct LlamaHfModel {
  HfLlamaConfig config;
  LlamaHfModelWeights weights;
};

absl::StatusOr<std::unique_ptr<LlamaHfModel>> LoadLlamaHfModel(
    const std::string& model_dir);

}  // namespace model
