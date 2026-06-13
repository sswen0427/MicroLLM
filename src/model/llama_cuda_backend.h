#pragma once

#include <unordered_map>

#include "model/llama_cpu_backend.h"

namespace model {

class CudaLlamaBackend final : public LlamaBackend {
public:
  base::DeviceType device_type() const override;

private:
  void Embedding(const tensor::Tensor &weight, int32_t token_id,
                 std::vector<float> &output) const override;
  tensor::Tensor EmbeddingTensor(const tensor::Tensor &weight,
                                 int32_t token_id) const override;
  void RmsNorm(const std::vector<float> &input, const tensor::Tensor &weight,
               double eps, std::vector<float> &output) const override;
  tensor::Tensor RmsNormTensor(const tensor::Tensor &input,
                               const tensor::Tensor &weight,
                               double eps) const override;
  void MatVec(const tensor::Tensor &weight, const std::vector<float> &input,
              std::vector<float> &output) const override;
  tensor::Tensor MatVecTensor(const tensor::Tensor &weight,
                              const tensor::Tensor &input) const override;
  void ApplyRopeToHeads(std::vector<float> &values, int32_t head_count,
                        int32_t head_size, int32_t position,
                        double rope_theta) const override;
  void StoreKvCache(const std::vector<float> &key,
                    const std::vector<float> &value, int32_t position,
                    int32_t max_position, int32_t kv_dim,
                    std::vector<float> &key_cache,
                    std::vector<float> &value_cache) const override;
  void AttentionWithCache(const std::vector<float> &query,
                          const std::vector<float> &key_cache,
                          const std::vector<float> &value_cache,
                          int32_t position, int32_t head_count,
                          int32_t head_size, int32_t kv_dim, int32_t kv_mul,
                          std::vector<float> &output) const override;
  void SwiGlu(const std::vector<float> &gate, const std::vector<float> &up,
              std::vector<float> &output) const override;
  tensor::Tensor SwiGluTensor(const tensor::Tensor &gate,
                              const tensor::Tensor &up) const override;
  void AddInPlace(std::vector<float> &left,
                  const std::vector<float> &right) const override;
  void AddInPlaceTensor(tensor::Tensor &left,
                        const tensor::Tensor &right) const override;
  int32_t ArgMaxToken(const tensor::Tensor &logits) const override;
  const tensor::Tensor &Fp32CudaWeight(const tensor::Tensor &weight) const;

  CpuLlamaBackend cpu_;
  mutable std::unordered_map<const tensor::Tensor *, tensor::Tensor>
      fp32_cuda_weights_;
};

} // namespace model
