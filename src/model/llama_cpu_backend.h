#pragma once

#include "model/llama_backend.h"

namespace model {

class CpuLlamaBackend final : public LlamaBackend {
 public:
  base::DeviceType device_type() const override;
  absl::StatusOr<LlamaForwardResult> ForwardToken(
      const LlamaHfModel &model, LlamaForwardState &state, int32_t token_id,
      int32_t position) const override;

 private:
  friend class CudaLlamaBackend;

  void Embedding(const tensor::Tensor &weight, int32_t token_id,
                 std::vector<float> &output) const;
  tensor::Tensor EmbeddingTensor(const tensor::Tensor &weight,
                                 int32_t token_id) const;
  void RmsNorm(const std::vector<float> &input, const tensor::Tensor &weight,
               double eps, std::vector<float> &output) const;
  tensor::Tensor RmsNormTensor(const tensor::Tensor &input,
                               const tensor::Tensor &weight, double eps) const;
  void MatVec(const tensor::Tensor &weight, const std::vector<float> &input,
              std::vector<float> &output) const;
  tensor::Tensor MatVecTensor(const tensor::Tensor &weight,
                              const tensor::Tensor &input) const;
  void ApplyRopeToHeads(std::vector<float> &values, int32_t head_count,
                        int32_t head_size, int32_t position,
                        double rope_theta) const;
  void StoreKvCache(const std::vector<float> &key,
                    const std::vector<float> &value, int32_t position,
                    int32_t max_position, int32_t kv_dim,
                    std::vector<float> &key_cache,
                    std::vector<float> &value_cache) const;
  void AttentionWithCache(const std::vector<float> &query,
                          const std::vector<float> &key_cache,
                          const std::vector<float> &value_cache,
                          int32_t position, int32_t head_count,
                          int32_t head_size, int32_t kv_dim, int32_t kv_mul,
                          std::vector<float> &output) const;
  void SwiGlu(const std::vector<float> &gate, const std::vector<float> &up,
              std::vector<float> &output) const;
  tensor::Tensor SwiGluTensor(const tensor::Tensor &gate,
                              const tensor::Tensor &up) const;
  void AddInPlace(std::vector<float> &left,
                  const std::vector<float> &right) const;
  void AddInPlaceTensor(tensor::Tensor &left,
                        const tensor::Tensor &right) const;
  int32_t ArgMaxToken(const tensor::Tensor &logits) const;
  static void SoftmaxInPlace(std::vector<float> &values);
};

}  // namespace model
