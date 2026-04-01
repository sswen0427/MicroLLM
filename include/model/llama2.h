#ifndef MICROLLM_INCLUDE_MODEL_LLAMA2_H
#define MICROLLM_INCLUDE_MODEL_LLAMA2_H
#include <cstdint>

#include "model/llama2_config.h"
#include "model/model.h"
#include "op/layer.h"

struct LLamaRawModelData {
  int32_t fd = -1;
  std::size_t file_size = 0;
  std::size_t current_offset = 0;
  float* data = nullptr;
  float* weight_data = nullptr;

  ~LLamaRawModelData();

  void add_offset(std::size_t offset);

  const float* weight() const;

  bool weight_is_valid(std::size_t peek) const;
};

class LLamaModel : public Model {
 public:
  explicit LLamaModel(std::string token_path, std::string model_path);

  Status init() override;

  Tensor forward(const std::vector<int>& tokens, int start_pos) override;

 private:
  Status read_model_file();

  int32_t vocab_size_ = 0;
  LlamaModelConfig config_;
  std::unique_ptr<LLamaRawModelData> raw_data_ = nullptr;
  std::vector<ParamLayerFp32> embedding_layers_;
};

#endif  // MICROLLM_INCLUDE_MODEL_LLAMA2_H
