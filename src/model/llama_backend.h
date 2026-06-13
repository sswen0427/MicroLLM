#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "base/types.h"
#include "tensor/tensor.h"

namespace model {

class LlamaBackend {
 public:
  virtual ~LlamaBackend() = default;

  virtual base::DeviceType device_type() const = 0;

  virtual void Embedding(const tensor::Tensor& weight, int32_t token_id,
                         std::vector<float>& output) const = 0;
  virtual void RmsNorm(const std::vector<float>& input,
                       const tensor::Tensor& weight, double eps,
                       std::vector<float>& output) const = 0;
  virtual void MatVec(const tensor::Tensor& weight,
                      const std::vector<float>& input,
                      std::vector<float>& output) const = 0;
  virtual void ApplyRopeToHeads(std::vector<float>& values, int32_t head_count,
                                int32_t head_size, int32_t position,
                                double rope_theta) const = 0;
  virtual void StoreKvCache(const std::vector<float>& key,
                            const std::vector<float>& value, int32_t position,
                            int32_t max_position, int32_t kv_dim,
                            std::vector<float>& key_cache,
                            std::vector<float>& value_cache) const = 0;
  virtual void AttentionWithCache(const std::vector<float>& query,
                                  const std::vector<float>& key_cache,
                                  const std::vector<float>& value_cache,
                                  int32_t position, int32_t head_count,
                                  int32_t head_size, int32_t kv_dim,
                                  int32_t kv_mul,
                                  std::vector<float>& output) const = 0;
  virtual void SwiGlu(const std::vector<float>& gate,
                      const std::vector<float>& up,
                      std::vector<float>& output) const = 0;
  virtual void AddInPlace(std::vector<float>& left,
                          const std::vector<float>& right) const = 0;
  virtual int32_t ArgMaxToken(const tensor::Tensor& logits) const = 0;
};

std::unique_ptr<LlamaBackend> CreateLlamaBackend(base::DeviceType device_type);

}  // namespace model
