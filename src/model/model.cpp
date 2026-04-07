#include "model/model.h"

#include <bits/ranges_algo.h>

#include <utility>

namespace model {

Model::Model(const base::TokenizerType& tokenizer_type,
             const base::ModelType& model_type, std::string token_path,
             std::string model_path, bool is_quant_model)
    : tokenizer_type_(tokenizer_type),
      model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      is_quant_model_(is_quant_model) {}

base::Status Model::create_encode_layer() {
  // create token encode decode layer
  if (tokenizer_type_ == base::TokenizerType::kEncodeSpe) {
    encode_layer_ = std::make_unique<op::EncodeLayerBase>();
  }
}

base::Status Model::gen_model_from_file() {
  config_ = std::make_unique<TransformerConfig>();
}

}  // namespace model
