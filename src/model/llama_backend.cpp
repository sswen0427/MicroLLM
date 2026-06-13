#include "model/llama_backend.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <safetensors.hh>
#include <vector>

namespace model {
namespace {

float TensorElementAsFloat(const tensor::Tensor& tensor, size_t offset) {
  switch (tensor.data_type()) {
    case base::DataType::kDataTypeFp32:
      return tensor.data<float>()[offset];
    case base::DataType::kDataTypeFp16:
      return safetensors::fp16_to_float(tensor.data<uint16_t>()[offset]);
    case base::DataType::kDataTypeBf16:
      return safetensors::bfloat16_to_float(tensor.data<uint16_t>()[offset]);
    default:
      LOG(FATAL) << "Unsupported floating point data type: "
                 << static_cast<int>(tensor.data_type());
  }
  return 0.0f;
}

class CpuLlamaBackend final : public LlamaBackend {
 public:
  base::DeviceType device_type() const override {
    return base::DeviceType::kDeviceCPU;
  }

  void Embedding(const tensor::Tensor& weight, int32_t token_id,
                 std::vector<float>& output) const override {
    const int32_t cols = weight.get_dim(1);
    output.resize(cols);
    const size_t row_offset = static_cast<size_t>(token_id) * cols;
    for (int32_t col = 0; col < cols; ++col) {
      output[col] = TensorElementAsFloat(weight, row_offset + col);
    }
  }

  void RmsNorm(const std::vector<float>& input, const tensor::Tensor& weight,
               double eps, std::vector<float>& output) const override {
    float square_sum = 0.0f;
    for (const float value : input) {
      square_sum += value * value;
    }
    const float mean_square = square_sum / static_cast<float>(input.size());
    const float scale = 1.0f / std::sqrt(mean_square + static_cast<float>(eps));

    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      output[i] = input[i] * scale * TensorElementAsFloat(weight, i);
    }
  }

  void MatVec(const tensor::Tensor& weight, const std::vector<float>& input,
              std::vector<float>& output) const override {
    const int32_t rows = weight.get_dim(0);
    const int32_t cols = weight.get_dim(1);
    CHECK_EQ(static_cast<int32_t>(input.size()), cols);
    output.assign(rows, 0.0f);
    for (int32_t row = 0; row < rows; ++row) {
      float sum = 0.0f;
      const size_t row_offset = static_cast<size_t>(row) * cols;
      for (int32_t col = 0; col < cols; ++col) {
        sum += TensorElementAsFloat(weight, row_offset + col) * input[col];
      }
      output[row] = sum;
    }
  }

  void ApplyRopeToHeads(std::vector<float>& values, int32_t head_count,
                        int32_t head_size, int32_t position,
                        double rope_theta) const override {
    CHECK_EQ(static_cast<int32_t>(values.size()), head_count * head_size);
    CHECK_EQ(head_size % 2, 0);
    const int32_t half_head_size = head_size / 2;
    for (int32_t head = 0; head < head_count; ++head) {
      const int32_t head_offset = head * head_size;
      for (int32_t i = 0; i < half_head_size; ++i) {
        const float freq = 1.0f / std::pow(static_cast<float>(rope_theta),
                                           static_cast<float>(2 * i) /
                                               static_cast<float>(head_size));
        const float angle = static_cast<float>(position) * freq;
        const float cos_value = std::cos(angle);
        const float sin_value = std::sin(angle);

        const int32_t first = head_offset + i;
        const int32_t second = head_offset + i + half_head_size;
        const float x0 = values[first];
        const float x1 = values[second];
        values[first] = x0 * cos_value - x1 * sin_value;
        values[second] = x0 * sin_value + x1 * cos_value;
      }
    }
  }

  void StoreKvCache(const std::vector<float>& key,
                    const std::vector<float>& value, int32_t position,
                    int32_t max_position, int32_t kv_dim,
                    std::vector<float>& key_cache,
                    std::vector<float>& value_cache) const override {
    CHECK_EQ(static_cast<int32_t>(key.size()), kv_dim);
    CHECK_EQ(static_cast<int32_t>(value.size()), kv_dim);
    CHECK_GE(position, 0);
    CHECK_LT(position, max_position);
    const size_t offset = static_cast<size_t>(position) * kv_dim;
    std::copy(key.begin(), key.end(), key_cache.begin() + offset);
    std::copy(value.begin(), value.end(), value_cache.begin() + offset);
  }

  void AttentionWithCache(const std::vector<float>& query,
                          const std::vector<float>& key_cache,
                          const std::vector<float>& value_cache,
                          int32_t position, int32_t head_count,
                          int32_t head_size, int32_t kv_dim, int32_t kv_mul,
                          std::vector<float>& output) const override {
    CHECK_GE(position, 0);
    CHECK_EQ(static_cast<int32_t>(query.size()), head_count * head_size);
    output.assign(static_cast<size_t>(head_count) * head_size, 0.0f);

    std::vector<float> scores(static_cast<size_t>(position) + 1);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    for (int32_t head = 0; head < head_count; ++head) {
      const int32_t kv_head = head / kv_mul;
      const int32_t query_offset = head * head_size;
      const int32_t output_offset = head * head_size;

      for (int32_t token = 0; token <= position; ++token) {
        const int32_t cache_offset = token * kv_dim + kv_head * head_size;
        float score = 0.0f;
        for (int32_t i = 0; i < head_size; ++i) {
          score += query[query_offset + i] * key_cache[cache_offset + i];
        }
        scores[token] = score * scale;
      }
      SoftmaxInPlace(scores);

      for (int32_t token = 0; token <= position; ++token) {
        const int32_t cache_offset = token * kv_dim + kv_head * head_size;
        const float score = scores[token];
        for (int32_t i = 0; i < head_size; ++i) {
          output[output_offset + i] += score * value_cache[cache_offset + i];
        }
      }
    }
  }

  void SwiGlu(const std::vector<float>& gate, const std::vector<float>& up,
              std::vector<float>& output) const override {
    CHECK_EQ(gate.size(), up.size());
    output.resize(gate.size());
    for (size_t i = 0; i < gate.size(); ++i) {
      const float silu = gate[i] / (1.0f + std::exp(-gate[i]));
      output[i] = silu * up[i];
    }
  }

  void AddInPlace(std::vector<float>& left,
                  const std::vector<float>& right) const override {
    CHECK_EQ(left.size(), right.size());
    for (size_t i = 0; i < left.size(); ++i) {
      left[i] += right[i];
    }
  }

  int32_t ArgMaxToken(const tensor::Tensor& logits) const override {
    CHECK(logits.data_type() == base::DataType::kDataTypeFp32);
    const float* data = logits.data<float>();
    int32_t best = 0;
    float best_value = data[0];
    for (int32_t i = 1; i < static_cast<int32_t>(logits.size()); ++i) {
      if (data[i] > best_value) {
        best = i;
        best_value = data[i];
      }
    }
    return best;
  }

 private:
  static void SoftmaxInPlace(std::vector<float>& values) {
    CHECK(!values.empty());
    const float max_value = *std::max_element(values.begin(), values.end());
    float sum = 0.0f;
    for (float& value : values) {
      value = std::exp(value - max_value);
      sum += value;
    }
    for (float& value : values) {
      value /= sum;
    }
  }
};

}  // namespace

std::unique_ptr<LlamaBackend> CreateLlamaBackend(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return std::make_unique<CpuLlamaBackend>();
  }
  LOG(FATAL) << "Unsupported LLaMA backend device type: "
             << static_cast<int>(device_type);
  return nullptr;
}

}  // namespace model
