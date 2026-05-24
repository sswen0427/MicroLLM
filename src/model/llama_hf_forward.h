#pragma once

#include <absl/status/statusor.h>

#include <cstdint>
#include <vector>

#include "model/llama_hf_model_loader.h"
#include "tensor/tensor.h"

namespace model {

struct LlamaForwardResult {
  tensor::Tensor logits;
  int32_t next_token = -1;
};

class LlamaHfRuntime {
 public:
  explicit LlamaHfRuntime(const LlamaHfModel& model);

  absl::StatusOr<LlamaForwardResult> ForwardToken(int32_t token_id,
                                                  int32_t position);

 private:
  struct LayerCache {
    std::vector<float> key;
    std::vector<float> value;
  };

  const LlamaHfModel& model_;
  int32_t head_size_ = 0;
  int32_t kv_dim_ = 0;
  int32_t kv_mul_ = 0;
  std::vector<LayerCache> layer_caches_;
};

}  // namespace model
