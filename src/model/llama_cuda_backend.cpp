#include <absl/status/status.h>
#include <absl/strings/str_cat.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/profile.h"
#include "cuda/add_kernel.cuh"
#include "cuda/emb_kernel.cuh"
#include "cuda/matmul_kernel.cuh"
#include "cuda/rmsnorm_kernel.cuh"
#include "cuda/swiglu_kernel.cuh"
#include "model/llama_backend.h"
#include "model/llama_backend_util.h"

namespace model {
namespace {

std::vector<int32_t> TensorDims(const tensor::Tensor &tensor) {
  std::vector<int32_t> dims;
  dims.reserve(tensor.dims_size());
  for (int32_t i = 0; i < tensor.dims_size(); ++i) {
    dims.push_back(tensor.get_dim(i));
  }
  return dims;
}

tensor::Tensor CopyVectorToCudaTensor(const std::vector<float> &values) {
  tensor::Tensor tensor = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {static_cast<int32_t>(values.size())},
      base::DeviceType::kDeviceCPU);
  std::copy(values.begin(), values.end(), tensor.data<float>());
  tensor.to_cuda();
  return tensor;
}

tensor::Tensor EnsureCudaTensor(const tensor::Tensor &tensor) {
  if (tensor.device_type() == base::DeviceType::kDeviceCUDA) {
    return tensor;
  }
  tensor::Tensor cuda_tensor = tensor.clone();
  cuda_tensor.to_cuda();
  return cuda_tensor;
}

std::vector<float> CopyTensorToVector(tensor::Tensor tensor) {
  tensor.to_cpu();
  std::vector<float> values(tensor.size());
  std::copy(tensor.data<float>(), tensor.data<float>() + tensor.size(),
            values.begin());
  return values;
}

void RmsNormCpu(const std::vector<float> &input, const tensor::Tensor &weight,
                double eps, std::vector<float> &output) {
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

void ApplyRopeToHeadsCpu(std::vector<float> &values, int32_t head_count,
                         int32_t head_size, int32_t position,
                         double rope_theta) {
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

void StoreKvCacheCpu(const std::vector<float> &key,
                     const std::vector<float> &value, int32_t position,
                     int32_t max_position, int32_t kv_dim,
                     std::vector<float> &key_cache,
                     std::vector<float> &value_cache) {
  CHECK_EQ(static_cast<int32_t>(key.size()), kv_dim);
  CHECK_EQ(static_cast<int32_t>(value.size()), kv_dim);
  CHECK_GE(position, 0);
  CHECK_LT(position, max_position);
  const size_t offset = static_cast<size_t>(position) * kv_dim;
  std::copy(key.begin(), key.end(), key_cache.begin() + offset);
  std::copy(value.begin(), value.end(), value_cache.begin() + offset);
}

void SoftmaxInPlace(std::vector<float> &values) {
  CHECK(!values.empty());
  const float max_value = *std::max_element(values.begin(), values.end());
  float sum = 0.0f;
  for (float &value : values) {
    value = std::exp(value - max_value);
    sum += value;
  }
  for (float &value : values) {
    value /= sum;
  }
}

/**
 * Computes single-token causal attention using the existing K/V cache.
 *
 * For each query head:
 *   scores_t = dot(q_head, k_t) / sqrt(head_size), t in [0, position]
 *   probs = softmax(scores)
 *   output_head = sum_t probs_t * v_t
 *
 * Loop structure:
 *   head loop  : computes one output head at a time.
 *   token loop : first builds scores against cached keys, then mixes cached
 *                values with the softmax probabilities.
 *
 * GQA/MQA mapping:
 *   kv_head = head / kv_mul
 * so multiple query heads may share the same K/V cache head.
 */
void AttentionWithCacheCpu(const std::vector<float> &query,
                           const std::vector<float> &key_cache,
                           const std::vector<float> &value_cache,
                           int32_t position, int32_t head_count,
                           int32_t head_size, int32_t kv_dim, int32_t kv_mul,
                           std::vector<float> &output) {
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

tensor::Tensor EmbeddingTensor(const tensor::Tensor &fp32_cuda_weight,
                               int32_t token_id) {
  tensor::Tensor input = tensor::Tensor::allocate(
      base::DataType::kDataTypeInt32, {1}, base::DeviceType::kDeviceCPU);
  input.data<int32_t>()[0] = token_id;
  tensor::Tensor output_tensor = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {fp32_cuda_weight.get_dim(1)},
      base::DeviceType::kDeviceCUDA);
  kernel::emb_kernel_cu(input, fp32_cuda_weight, output_tensor,
                        fp32_cuda_weight.get_dim(0), nullptr);
  return output_tensor;
}

tensor::Tensor RmsNormTensor(const tensor::Tensor &input,
                             const tensor::Tensor &weight,
                             const tensor::Tensor &fp32_cuda_weight,
                             double eps) {
  if (std::abs(eps - 1e-5) > 1e-12) {
    std::vector<float> output;
    RmsNormCpu(CopyTensorToVector(input), weight, eps, output);
    return CopyVectorToCudaTensor(output);
  }

  tensor::Tensor input_tensor = EnsureCudaTensor(input);
  tensor::Tensor output_tensor =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32,
                               {static_cast<int32_t>(input_tensor.size())},
                               base::DeviceType::kDeviceCUDA);
  kernel::rmsnorm_kernel_cu(input_tensor, fp32_cuda_weight, output_tensor,
                            nullptr);
  return output_tensor;
}

tensor::Tensor MatVecTensor(const tensor::Tensor &weight,
                            const tensor::Tensor &fp32_cuda_weight,
                            const tensor::Tensor &input) {
  tensor::Tensor input_tensor = EnsureCudaTensor(input);
  tensor::Tensor output_tensor = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {weight.get_dim(0)},
      base::DeviceType::kDeviceCUDA);
  kernel::matmul_kernel_cu(input_tensor, fp32_cuda_weight, output_tensor, 1.0f,
                           nullptr);
  return output_tensor;
}

// SwiGLU is the gated activation used in LLaMA's FFN:
//   silu(x) = x / (1 + exp(-x))
//   output = silu(gate_proj(x)) * up_proj(x)
tensor::Tensor SwiGluTensor(const tensor::Tensor &gate,
                            const tensor::Tensor &up) {
  tensor::Tensor gate_tensor = EnsureCudaTensor(gate);
  tensor::Tensor up_tensor = EnsureCudaTensor(up);
  tensor::Tensor output_tensor = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {static_cast<int32_t>(gate.size())},
      base::DeviceType::kDeviceCUDA);
  kernel::swiglu_kernel_cu(gate_tensor, up_tensor, output_tensor, nullptr);
  return output_tensor;
}

void AddInPlaceTensor(tensor::Tensor &left, const tensor::Tensor &right) {
  if (left.device_type() == base::DeviceType::kDeviceCUDA ||
      right.device_type() == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor right_tensor = EnsureCudaTensor(right);
    if (left.device_type() == base::DeviceType::kDeviceCPU) {
      left.to_cuda();
    }
    kernel::add_inplace_kernel_cu(left, right_tensor, nullptr);
    return;
  }

  CHECK(left.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(right.device_type() == base::DeviceType::kDeviceCPU);
  CHECK_EQ(left.size(), right.size());
  for (size_t i = 0; i < left.size(); ++i) {
    left.data<float>()[i] += right.data<float>()[i];
  }
}

int32_t ArgMaxToken(const tensor::Tensor &logits) {
  CHECK(logits.data_type() == base::DataType::kDataTypeFp32);
  const float *data = logits.data<float>();
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

} // namespace

class CudaLlamaBackend final : public LlamaBackend {
public:
  base::DeviceType device_type() const override;
  absl::StatusOr<LlamaForwardResult>
  ForwardToken(const LlamaHfModel &model, LlamaForwardState &state,
               int32_t token_id, int32_t position) const override;

private:
  const tensor::Tensor &Fp32CudaWeight(const tensor::Tensor &weight) const;

  mutable std::unordered_map<const tensor::Tensor *, tensor::Tensor>
      fp32_cuda_weights_;
};

std::unique_ptr<LlamaBackend> CreateCudaLlamaBackend() {
  return std::make_unique<CudaLlamaBackend>();
}

base::DeviceType CudaLlamaBackend::device_type() const {
  return base::DeviceType::kDeviceCUDA;
}

absl::StatusOr<LlamaForwardResult>
CudaLlamaBackend::ForwardToken(const LlamaHfModel &model,
                               LlamaForwardState &state, int32_t token_id,
                               int32_t position) const {
  const HfLlamaConfig &config = model.config;
  if (token_id < 0 || token_id >= config.vocab_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("token_id is out of range: ", token_id,
                     ", vocab_size=", config.vocab_size));
  }
  if (position < 0 || position >= config.max_position_embeddings) {
    return absl::InvalidArgumentError(absl::StrCat(
        "position is out of range: ", position,
        ", max_position_embeddings=", config.max_position_embeddings));
  }

  LOG(INFO) << "start LLaMA CUDA one-token forward: token_id=" << token_id
            << ", position=" << position;
  state.profile.forward_calls += 1;

  tensor::Tensor hidden_state;
  {
    base::ScopedProfile profile(state.profile.embedding_ms);
    hidden_state = EmbeddingTensor(
        Fp32CudaWeight(model.weights.token_embedding), token_id);
  }

  tensor::Tensor norm;
  tensor::Tensor query_tensor;
  tensor::Tensor key_tensor;
  tensor::Tensor value_tensor;
  tensor::Tensor projected_attention;
  tensor::Tensor gate;
  tensor::Tensor up;
  tensor::Tensor activated;
  tensor::Tensor projected_ffn;
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> attention_output;

  for (int32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    const LlamaHfLayerWeights &weights = model.weights.layers[layer];

    {
      base::ScopedProfile profile(state.profile.attention_norm_ms);
      norm = RmsNormTensor(hidden_state, weights.input_layernorm,
                           Fp32CudaWeight(weights.input_layernorm),
                           config.rms_norm_eps);
    }
    {
      base::ScopedProfile profile(state.profile.qkv_proj_ms);
      query_tensor =
          MatVecTensor(weights.q_proj, Fp32CudaWeight(weights.q_proj), norm);
      key_tensor =
          MatVecTensor(weights.k_proj, Fp32CudaWeight(weights.k_proj), norm);
      value_tensor =
          MatVecTensor(weights.v_proj, Fp32CudaWeight(weights.v_proj), norm);
      query = CopyTensorToVector(query_tensor);
      key = CopyTensorToVector(key_tensor);
      value = CopyTensorToVector(value_tensor);
    }
    {
      base::ScopedProfile profile(state.profile.rope_ms);
      ApplyRopeToHeadsCpu(query, config.num_attention_heads, state.head_size,
                          position, config.rope_theta);
      ApplyRopeToHeadsCpu(key, config.num_key_value_heads, state.head_size,
                          position, config.rope_theta);
    }
    {
      base::ScopedProfile profile(state.profile.kv_cache_ms);
      StoreKvCacheCpu(key, value, position, config.max_position_embeddings,
                      state.kv_dim, state.layer_caches[layer].key,
                      state.layer_caches[layer].value);
    }
    {
      base::ScopedProfile profile(state.profile.attention_ms);
      AttentionWithCacheCpu(query, state.layer_caches[layer].key,
                            state.layer_caches[layer].value, position,
                            config.num_attention_heads, state.head_size,
                            state.kv_dim, state.kv_mul, attention_output);
    }
    {
      base::ScopedProfile profile(state.profile.attention_output_proj_ms);
      projected_attention =
          MatVecTensor(weights.o_proj, Fp32CudaWeight(weights.o_proj),
                       CopyVectorToCudaTensor(attention_output));
    }
    {
      base::ScopedProfile profile(state.profile.attention_residual_ms);
      AddInPlaceTensor(hidden_state, projected_attention);
    }
    {
      base::ScopedProfile profile(state.profile.ffn_norm_ms);
      norm = RmsNormTensor(hidden_state, weights.post_attention_layernorm,
                           Fp32CudaWeight(weights.post_attention_layernorm),
                           config.rms_norm_eps);
    }
    {
      base::ScopedProfile profile(state.profile.ffn_up_gate_proj_ms);
      gate = MatVecTensor(weights.gate_proj, Fp32CudaWeight(weights.gate_proj),
                          norm);
      up = MatVecTensor(weights.up_proj, Fp32CudaWeight(weights.up_proj), norm);
    }
    {
      base::ScopedProfile profile(state.profile.swiglu_ms);
      activated = SwiGluTensor(gate, up);
    }
    {
      base::ScopedProfile profile(state.profile.ffn_down_proj_ms);
      projected_ffn = MatVecTensor(
          weights.down_proj, Fp32CudaWeight(weights.down_proj), activated);
    }
    {
      base::ScopedProfile profile(state.profile.ffn_residual_ms);
      AddInPlaceTensor(hidden_state, projected_ffn);
    }
  }

  {
    base::ScopedProfile profile(state.profile.final_norm_ms);
    norm = RmsNormTensor(hidden_state, model.weights.final_norm,
                         Fp32CudaWeight(model.weights.final_norm),
                         config.rms_norm_eps);
  }

  LlamaForwardResult result;
  {
    base::ScopedProfile profile(state.profile.lm_head_ms);
    result.logits = MatVecTensor(model.weights.lm_head,
                                 Fp32CudaWeight(model.weights.lm_head), norm);
  }
  result.logits.to_cpu();
  CHECK_EQ(static_cast<int32_t>(result.logits.size()), config.vocab_size);
  {
    base::ScopedProfile profile(state.profile.argmax_ms);
    result.next_token = ArgMaxToken(result.logits);
  }

  LOG(INFO) << "finish LLaMA CUDA one-token forward: next_token="
            << result.next_token;
  return result;
}

const tensor::Tensor &
CudaLlamaBackend::Fp32CudaWeight(const tensor::Tensor &weight) const {
  const auto cached = fp32_cuda_weights_.find(&weight);
  if (cached != fp32_cuda_weights_.end()) {
    return cached->second;
  }

  tensor::Tensor fp32_weight;
  if (weight.data_type() == base::DataType::kDataTypeFp32) {
    fp32_weight = weight.clone();
    if (fp32_weight.device_type() == base::DeviceType::kDeviceCPU) {
      fp32_weight.to_cuda();
    }
  } else {
    fp32_weight = tensor::Tensor::allocate(base::DataType::kDataTypeFp32,
                                           TensorDims(weight),
                                           base::DeviceType::kDeviceCPU);
    for (size_t i = 0; i < weight.size(); ++i) {
      fp32_weight.data<float>()[i] = TensorElementAsFloat(weight, i);
    }
    fp32_weight.to_cuda();
  }

  auto insert_result =
      fp32_cuda_weights_.emplace(&weight, std::move(fp32_weight));
  return insert_result.first->second;
}

} // namespace model
