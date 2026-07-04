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
#include "cuda/add.cuh"
#include "cuda/embedding.cuh"
#include "cuda/llama.cuh"
#include "cuda/matmul.cuh"
#include "cuda/rmsnorm.cuh"
#include "cuda/swiglu.cuh"
#include "model/llama_backend.h"
#include "model/llama_backend_util.h"

namespace model {
namespace {

LlamaForwardState CreateCudaForwardState(const HfLlamaConfig &config) {
  LlamaForwardState state;
  if (config.num_attention_heads > 0) {
    state.head_size = config.hidden_size / config.num_attention_heads;
  }
  state.kv_dim = config.num_key_value_heads * state.head_size;
  if (config.num_key_value_heads > 0) {
    state.kv_mul = config.num_attention_heads / config.num_key_value_heads;
  }

  state.kv_cache = LlamaKvCache::Allocate(config, state.kv_dim,
                                          base::DeviceType::kDeviceCUDA);
  return state;
}

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

tensor::Tensor EmbeddingTensor(const tensor::Tensor &fp32_cuda_weight,
                               int32_t token_id) {
  tensor::Tensor input = tensor::Tensor::allocate(
      base::DataType::kDataTypeInt32, {1}, base::DeviceType::kDeviceCPU);
  input.data<int32_t>()[0] = token_id;
  input.to_cuda();
  tensor::Tensor output_tensor = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {fp32_cuda_weight.get_dim(1)},
      base::DeviceType::kDeviceCUDA);
  kernel::EmbeddingCuda(input, fp32_cuda_weight, output_tensor,
                        fp32_cuda_weight.get_dim(0), nullptr);
  return output_tensor;
}

tensor::Tensor EmbeddingTokensTensor(const tensor::Tensor &fp32_cuda_weight,
                                     const std::vector<int32_t> &token_ids) {
  tensor::Tensor input = tensor::Tensor::allocate(
      base::DataType::kDataTypeInt32, {static_cast<int32_t>(token_ids.size())},
      base::DeviceType::kDeviceCPU);
  std::copy(token_ids.begin(), token_ids.end(), input.data<int32_t>());
  input.to_cuda();

  tensor::Tensor output_tensor = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32,
      {static_cast<int32_t>(token_ids.size()), fp32_cuda_weight.get_dim(1)},
      base::DeviceType::kDeviceCUDA);
  kernel::EmbeddingCuda(input, fp32_cuda_weight, output_tensor,
                        fp32_cuda_weight.get_dim(0), nullptr);
  return output_tensor;
}

tensor::Tensor RmsNormTensor(const tensor::Tensor &input,
                             const tensor::Tensor &weight,
                             const tensor::Tensor &fp32_cuda_weight,
                             double eps) {
  if (input.dims_size() == 2) {
    tensor::Tensor input_tensor = EnsureCudaTensor(input);
    tensor::Tensor output_tensor = tensor::Tensor::allocate(
        base::DataType::kDataTypeFp32,
        {input_tensor.get_dim(0), input_tensor.get_dim(1)},
        base::DeviceType::kDeviceCUDA);
    kernel::RmsNormBatchCuda(input_tensor, fp32_cuda_weight, output_tensor,
                             nullptr, static_cast<float>(eps));
    return output_tensor;
  }

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
  kernel::RmsNormCuda(input_tensor, fp32_cuda_weight, output_tensor, nullptr);
  return output_tensor;
}

tensor::Tensor MatVecTensor(const tensor::Tensor &weight,
                            const tensor::Tensor &fp32_cuda_weight,
                            const tensor::Tensor &input) {
  tensor::Tensor input_tensor = EnsureCudaTensor(input);
  if (input_tensor.dims_size() == 2) {
    tensor::Tensor output_tensor = tensor::Tensor::allocate(
        base::DataType::kDataTypeFp32,
        {input_tensor.get_dim(0), weight.get_dim(0)},
        base::DeviceType::kDeviceCUDA);
    kernel::MatmulBatchCuda(input_tensor, fp32_cuda_weight, output_tensor, 1.0f,
                            nullptr);
    return output_tensor;
  }

  tensor::Tensor output_tensor = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {weight.get_dim(0)},
      base::DeviceType::kDeviceCUDA);
  kernel::MatmulCuda(input_tensor, fp32_cuda_weight, output_tensor, 1.0f,
                     nullptr);
  return output_tensor;
}

tensor::Tensor SwiGluTensor(const tensor::Tensor &gate,
                            const tensor::Tensor &up) {
  tensor::Tensor gate_tensor = EnsureCudaTensor(gate);
  tensor::Tensor up_tensor = EnsureCudaTensor(up);
  tensor::Tensor output_tensor = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {static_cast<int32_t>(gate.size())},
      base::DeviceType::kDeviceCUDA);
  kernel::SwiGluCuda(gate_tensor, up_tensor, output_tensor, nullptr);
  return output_tensor;
}

void AddInPlaceTensor(tensor::Tensor &left, const tensor::Tensor &right) {
  if (left.device_type() == base::DeviceType::kDeviceCUDA ||
      right.device_type() == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor right_tensor = EnsureCudaTensor(right);
    if (left.device_type() == base::DeviceType::kDeviceCPU) {
      left.to_cuda();
    }
    kernel::AddInPlaceCuda(left, right_tensor, nullptr);
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

tensor::Tensor LastLogitsToCpu(tensor::Tensor logits) {
  CHECK(logits.data_type() == base::DataType::kDataTypeFp32);
  if (logits.dims_size() == 1) {
    logits.to_cpu();
    return logits;
  }

  CHECK_EQ(logits.dims_size(), 2);
  const int32_t seq_len = logits.get_dim(0);
  const int32_t vocab_size = logits.get_dim(1);
  logits.to_cpu();
  tensor::Tensor last_logits = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {vocab_size}, base::DeviceType::kDeviceCPU);
  const float *src =
      logits.data<float>() + static_cast<size_t>(seq_len - 1) * vocab_size;
  std::copy(src, src + vocab_size, last_logits.data<float>());
  return last_logits;
}

class CudaLlamaBackend final : public LlamaBackend {
 public:
  explicit CudaLlamaBackend(const HfLlamaConfig &config);

  base::DeviceType device_type() const override;

  absl::StatusOr<LlamaForwardResult> Forward(
      const LlamaHfModel &model, const std::vector<int32_t> &token_ids,
      int32_t start_position) override;

  const LlamaForwardProfile &profile() const override;

 private:
  absl::StatusOr<LlamaForwardResult> Decode(const LlamaHfModel &model,
                                            int32_t token_id,
                                            int32_t position);
  absl::StatusOr<LlamaForwardResult> Prefill(
      const LlamaHfModel &model, const std::vector<int32_t> &token_ids,
      int32_t start_position);

  const tensor::Tensor &Fp32CudaWeight(const tensor::Tensor &weight);

  LlamaForwardState forward_state_;

  std::unordered_map<const tensor::Tensor *, tensor::Tensor> fp32_cuda_weights_;
};

CudaLlamaBackend::CudaLlamaBackend(const HfLlamaConfig &config)
    : forward_state_(CreateCudaForwardState(config)) {}

base::DeviceType CudaLlamaBackend::device_type() const {
  return base::DeviceType::kDeviceCUDA;
}

absl::StatusOr<LlamaForwardResult> CudaLlamaBackend::Forward(
    const LlamaHfModel &model, const std::vector<int32_t> &token_ids,
    int32_t start_position) {
  if (token_ids.empty()) {
    return absl::InvalidArgumentError("forward token_ids must not be empty.");
  }
  forward_state_.profile.forward_calls += 1;
  const bool is_decode = token_ids.size() == 1 && start_position > 0;
  if (is_decode) {
    forward_state_.profile.decode_calls += 1;
    forward_state_.profile.decode_tokens += 1;
    return Decode(model, token_ids.front(), start_position);
  }

  forward_state_.profile.prefill_calls += 1;
  forward_state_.profile.prefill_tokens +=
      static_cast<int64_t>(token_ids.size());
  return Prefill(model, token_ids, start_position);
}

absl::StatusOr<LlamaForwardResult> CudaLlamaBackend::Decode(
    const LlamaHfModel &model, int32_t token_id, int32_t position) {
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
  forward_state_.kv_cache.ValidateWritePosition(position);

  tensor::Tensor hidden_state;
  {
    base::ScopedProfile profile(forward_state_.profile.embedding_ms);
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
  tensor::Tensor attention_output_tensor;

  for (int32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    const LlamaHfLayerWeights &weights = model.weights.layers[layer];

    {
      base::ScopedProfile profile(forward_state_.profile.attention_norm_ms);
      norm = RmsNormTensor(hidden_state, weights.input_layernorm,
                           Fp32CudaWeight(weights.input_layernorm),
                           config.rms_norm_eps);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.qkv_proj_ms);
      query_tensor =
          MatVecTensor(weights.q_proj, Fp32CudaWeight(weights.q_proj), norm);
      key_tensor =
          MatVecTensor(weights.k_proj, Fp32CudaWeight(weights.k_proj), norm);
      value_tensor =
          MatVecTensor(weights.v_proj, Fp32CudaWeight(weights.v_proj), norm);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.rope_ms);
      kernel::RopeInPlaceCuda(query_tensor, config.num_attention_heads,
                              forward_state_.head_size, position,
                              config.rope_theta);
      kernel::RopeInPlaceCuda(key_tensor, config.num_key_value_heads,
                              forward_state_.head_size, position,
                              config.rope_theta);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.kv_cache_ms);
      kernel::StoreKvCacheCuda(
          key_tensor, value_tensor, forward_state_.kv_cache.key(layer),
          forward_state_.kv_cache.value(layer), position,
          forward_state_.kv_dim);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.attention_ms);
      attention_output_tensor = tensor::Tensor::allocate(
          base::DataType::kDataTypeFp32,
          {config.num_attention_heads * forward_state_.head_size},
          base::DeviceType::kDeviceCUDA);
      kernel::AttentionWithCacheCuda(
          query_tensor, forward_state_.kv_cache.key(layer),
          forward_state_.kv_cache.value(layer), attention_output_tensor,
          position, config.num_attention_heads, forward_state_.head_size,
          forward_state_.kv_dim, forward_state_.kv_mul);
    }
    {
      base::ScopedProfile profile(
          forward_state_.profile.attention_output_proj_ms);
      projected_attention =
          MatVecTensor(weights.o_proj, Fp32CudaWeight(weights.o_proj),
                       attention_output_tensor);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.attention_residual_ms);
      AddInPlaceTensor(hidden_state, projected_attention);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.ffn_norm_ms);
      norm = RmsNormTensor(hidden_state, weights.post_attention_layernorm,
                           Fp32CudaWeight(weights.post_attention_layernorm),
                           config.rms_norm_eps);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.ffn_up_gate_proj_ms);
      gate = MatVecTensor(weights.gate_proj, Fp32CudaWeight(weights.gate_proj),
                          norm);
      up = MatVecTensor(weights.up_proj, Fp32CudaWeight(weights.up_proj), norm);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.swiglu_ms);
      activated = SwiGluTensor(gate, up);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.ffn_down_proj_ms);
      projected_ffn = MatVecTensor(
          weights.down_proj, Fp32CudaWeight(weights.down_proj), activated);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.ffn_residual_ms);
      AddInPlaceTensor(hidden_state, projected_ffn);
    }
  }
  forward_state_.kv_cache.CommitToken(position);

  {
    base::ScopedProfile profile(forward_state_.profile.final_norm_ms);
    norm = RmsNormTensor(hidden_state, model.weights.final_norm,
                         Fp32CudaWeight(model.weights.final_norm),
                         config.rms_norm_eps);
  }

  LlamaForwardResult result;
  {
    base::ScopedProfile profile(forward_state_.profile.lm_head_ms);
    result.logits = MatVecTensor(model.weights.lm_head,
                                 Fp32CudaWeight(model.weights.lm_head), norm);
  }
  result.logits.to_cpu();
  CHECK_EQ(static_cast<int32_t>(result.logits.size()), config.vocab_size);
  {
    base::ScopedProfile profile(forward_state_.profile.argmax_ms);
    result.next_token = ArgMaxToken(result.logits);
  }

  LOG(INFO) << "finish LLaMA CUDA one-token forward: next_token="
            << result.next_token;
  return result;
}

absl::StatusOr<LlamaForwardResult> CudaLlamaBackend::Prefill(
    const LlamaHfModel &model, const std::vector<int32_t> &token_ids,
    int32_t start_position) {
  const HfLlamaConfig &config = model.config;
  if (token_ids.empty()) {
    return absl::InvalidArgumentError("forward token_ids must not be empty.");
  }
  if (start_position < 0 ||
      start_position + static_cast<int32_t>(token_ids.size()) >
          config.max_position_embeddings) {
    return absl::InvalidArgumentError(absl::StrCat(
        "prefill positions are out of range: start_position=", start_position,
        ", token_count=", token_ids.size(),
        ", max_position_embeddings=", config.max_position_embeddings));
  }
  for (const int32_t token_id : token_ids) {
    if (token_id < 0 || token_id >= config.vocab_size) {
      return absl::InvalidArgumentError(
          absl::StrCat("token_id is out of range: ", token_id,
                       ", vocab_size=", config.vocab_size));
    }
  }

  LOG(INFO) << "start LLaMA CUDA prefill: token_count=" << token_ids.size()
            << ", start_position=" << start_position;
  forward_state_.kv_cache.ValidateWriteRange(
      start_position, static_cast<int32_t>(token_ids.size()));

  tensor::Tensor hidden_state;
  {
    base::ScopedProfile profile(forward_state_.profile.embedding_ms);
    hidden_state = EmbeddingTokensTensor(
        Fp32CudaWeight(model.weights.token_embedding), token_ids);
  }

  tensor::Tensor norm;
  tensor::Tensor query_tensor;
  tensor::Tensor key_tensor;
  tensor::Tensor value_tensor;
  tensor::Tensor attention_output_tensor;
  tensor::Tensor projected_attention;
  tensor::Tensor gate;
  tensor::Tensor up;
  tensor::Tensor activated;
  tensor::Tensor projected_ffn;

  for (int32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    const LlamaHfLayerWeights &weights = model.weights.layers[layer];

    {
      base::ScopedProfile profile(forward_state_.profile.attention_norm_ms);
      norm = RmsNormTensor(hidden_state, weights.input_layernorm,
                           Fp32CudaWeight(weights.input_layernorm),
                           config.rms_norm_eps);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.qkv_proj_ms);
      query_tensor =
          MatVecTensor(weights.q_proj, Fp32CudaWeight(weights.q_proj), norm);
      key_tensor =
          MatVecTensor(weights.k_proj, Fp32CudaWeight(weights.k_proj), norm);
      value_tensor =
          MatVecTensor(weights.v_proj, Fp32CudaWeight(weights.v_proj), norm);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.rope_ms);
      kernel::RopeInPlaceBatchCuda(query_tensor, config.num_attention_heads,
                                   forward_state_.head_size, start_position,
                                   config.rope_theta);
      kernel::RopeInPlaceBatchCuda(key_tensor, config.num_key_value_heads,
                                   forward_state_.head_size, start_position,
                                   config.rope_theta);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.kv_cache_ms);
      kernel::StoreKvCacheBatchCuda(
          key_tensor, value_tensor, forward_state_.kv_cache.key(layer),
          forward_state_.kv_cache.value(layer), start_position,
          forward_state_.kv_dim);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.attention_ms);
      attention_output_tensor = tensor::Tensor::allocate(
          base::DataType::kDataTypeFp32,
          {static_cast<int32_t>(token_ids.size()),
           config.num_attention_heads * forward_state_.head_size},
          base::DeviceType::kDeviceCUDA);
      kernel::AttentionWithCacheBatchCuda(
          query_tensor, forward_state_.kv_cache.key(layer),
          forward_state_.kv_cache.value(layer), attention_output_tensor,
          start_position, config.num_attention_heads, forward_state_.head_size,
          forward_state_.kv_dim, forward_state_.kv_mul);
    }
    {
      base::ScopedProfile profile(
          forward_state_.profile.attention_output_proj_ms);
      projected_attention =
          MatVecTensor(weights.o_proj, Fp32CudaWeight(weights.o_proj),
                       attention_output_tensor);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.attention_residual_ms);
      AddInPlaceTensor(hidden_state, projected_attention);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.ffn_norm_ms);
      norm = RmsNormTensor(hidden_state, weights.post_attention_layernorm,
                           Fp32CudaWeight(weights.post_attention_layernorm),
                           config.rms_norm_eps);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.ffn_up_gate_proj_ms);
      gate = MatVecTensor(weights.gate_proj, Fp32CudaWeight(weights.gate_proj),
                          norm);
      up = MatVecTensor(weights.up_proj, Fp32CudaWeight(weights.up_proj), norm);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.swiglu_ms);
      activated = SwiGluTensor(gate, up);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.ffn_down_proj_ms);
      projected_ffn = MatVecTensor(
          weights.down_proj, Fp32CudaWeight(weights.down_proj), activated);
    }
    {
      base::ScopedProfile profile(forward_state_.profile.ffn_residual_ms);
      AddInPlaceTensor(hidden_state, projected_ffn);
    }
  }
  forward_state_.kv_cache.CommitTokens(
      start_position, static_cast<int32_t>(token_ids.size()));

  {
    base::ScopedProfile profile(forward_state_.profile.final_norm_ms);
    norm = RmsNormTensor(hidden_state, model.weights.final_norm,
                         Fp32CudaWeight(model.weights.final_norm),
                         config.rms_norm_eps);
  }

  LlamaForwardResult result;
  {
    base::ScopedProfile profile(forward_state_.profile.lm_head_ms);
    result.logits = MatVecTensor(model.weights.lm_head,
                                 Fp32CudaWeight(model.weights.lm_head), norm);
  }
  result.logits = LastLogitsToCpu(std::move(result.logits));
  CHECK_EQ(static_cast<int32_t>(result.logits.size()), config.vocab_size);
  {
    base::ScopedProfile profile(forward_state_.profile.argmax_ms);
    result.next_token = ArgMaxToken(result.logits);
  }

  LOG(INFO) << "finish LLaMA CUDA prefill: next_token=" << result.next_token;
  return result;
}

const LlamaForwardProfile &CudaLlamaBackend::profile() const {
  return forward_state_.profile;
}

const tensor::Tensor &CudaLlamaBackend::Fp32CudaWeight(
    const tensor::Tensor &weight) {
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

}  // namespace

std::unique_ptr<LlamaBackend> CreateCudaLlamaBackend(
    const HfLlamaConfig &config) {
  return std::make_unique<CudaLlamaBackend>(config);
}

}  // namespace model
