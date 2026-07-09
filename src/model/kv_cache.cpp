#include "model/kv_cache.h"

#include <glog/logging.h>

#include <algorithm>

namespace model {

KvCache KvCache::Allocate(int32_t layer_count, int32_t max_seq_len,
                          int32_t kv_dim, base::DeviceType device_type) {
  KvCache cache;
  cache.max_seq_len_ = max_seq_len;
  cache.kv_dim_ = kv_dim;
  cache.device_type_ = device_type;

  if (layer_count <= 0 || cache.max_seq_len_ <= 0 || cache.kv_dim_ <= 0) {
    return cache;
  }

  cache.layers_.resize(layer_count);
  for (LayerCache& layer : cache.layers_) {
    layer.key = tensor::Tensor::allocate(base::DataType::kDataTypeFp32,
                                         {cache.max_seq_len_, cache.kv_dim_},
                                         device_type);
    layer.value = tensor::Tensor::allocate(base::DataType::kDataTypeFp32,
                                           {cache.max_seq_len_, cache.kv_dim_},
                                           device_type);
  }
  return cache;
}

bool KvCache::empty() const { return layers_.empty(); }

int32_t KvCache::layer_count() const {
  return static_cast<int32_t>(layers_.size());
}

int32_t KvCache::max_seq_len() const { return max_seq_len_; }

int32_t KvCache::kv_dim() const { return kv_dim_; }

int32_t KvCache::seq_len() const { return seq_len_; }

base::DeviceType KvCache::device_type() const { return device_type_; }

tensor::Tensor& KvCache::key(int32_t layer) {
  CHECK_GE(layer, 0);
  CHECK_LT(layer, layer_count());
  return layers_[layer].key;
}

tensor::Tensor& KvCache::value(int32_t layer) {
  CHECK_GE(layer, 0);
  CHECK_LT(layer, layer_count());
  return layers_[layer].value;
}

const tensor::Tensor& KvCache::key(int32_t layer) const {
  CHECK_GE(layer, 0);
  CHECK_LT(layer, layer_count());
  return layers_[layer].key;
}

const tensor::Tensor& KvCache::value(int32_t layer) const {
  CHECK_GE(layer, 0);
  CHECK_LT(layer, layer_count());
  return layers_[layer].value;
}

void KvCache::ValidateWritePosition(int32_t position) const {
  CHECK_GE(position, 0);
  CHECK_LT(position, max_seq_len_);
  CHECK_LE(position, seq_len_)
      << "KV cache writes must be sequential. position=" << position
      << ", seq_len=" << seq_len_;
}

void KvCache::ValidateWriteRange(int32_t start_position,
                                 int32_t token_count) const {
  CHECK_GT(token_count, 0);
  CHECK_GE(start_position, 0);
  CHECK_EQ(start_position, seq_len_)
      << "KV cache writes must append to the current sequence. start_position="
      << start_position << ", seq_len=" << seq_len_;
  CHECK_LE(start_position + token_count, max_seq_len_);
}

void KvCache::CommitToken(int32_t position) {
  ValidateWritePosition(position);
  seq_len_ = std::max(seq_len_, position + 1);
}

void KvCache::CommitTokens(int32_t start_position, int32_t token_count) {
  ValidateWriteRange(start_position, token_count);
  seq_len_ = start_position + token_count;
}

void KvCache::Reset() { seq_len_ = 0; }

}  // namespace model
