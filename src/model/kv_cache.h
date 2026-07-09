#pragma once

#include <cstdint>
#include <vector>

#include "base/types.h"
#include "tensor/tensor.h"

namespace model {

class KvCache {
 public:
  KvCache() = default;

  static KvCache Allocate(int32_t layer_count, int32_t max_seq_len,
                          int32_t kv_dim, base::DeviceType device_type);

  [[nodiscard]] bool empty() const;
  [[nodiscard]] int32_t layer_count() const;
  [[nodiscard]] int32_t max_seq_len() const;
  [[nodiscard]] int32_t kv_dim() const;
  [[nodiscard]] int32_t seq_len() const;
  [[nodiscard]] base::DeviceType device_type() const;

  tensor::Tensor& key(int32_t layer);
  tensor::Tensor& value(int32_t layer);
  const tensor::Tensor& key(int32_t layer) const;
  const tensor::Tensor& value(int32_t layer) const;

  void ValidateWritePosition(int32_t position) const;
  void ValidateWriteRange(int32_t start_position, int32_t token_count) const;
  void CommitToken(int32_t position);
  void CommitTokens(int32_t start_position, int32_t token_count);
  void Reset();

 private:
  struct LayerCache {
    tensor::Tensor key;
    tensor::Tensor value;
  };

  std::vector<LayerCache> layers_;
  int32_t max_seq_len_ = 0;
  int32_t kv_dim_ = 0;
  int32_t seq_len_ = 0;
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
};

}  // namespace model
