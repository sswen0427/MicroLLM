#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "base/types.h"
#include "model/llama_backend.h"
#include "model/llama_hf_model_loader.h"
#include "tensor/tensor.h"

namespace {

tensor::Tensor MakeFp32Tensor(const std::vector<int32_t> &dims,
                              const std::vector<float> &values) {
  tensor::Tensor tensor = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, dims, base::DeviceType::kDeviceCPU);
  EXPECT_EQ(tensor.size(), values.size());
  std::copy(values.begin(), values.end(), tensor.data<float>());
  return tensor;
}

tensor::Tensor Zeros(const std::vector<int32_t> &dims, size_t size) {
  return MakeFp32Tensor(dims, std::vector<float>(size, 0.0f));
}

model::LlamaHfModel MakeTinyForwardModel() {
  model::LlamaHfModel model;
  model.config.attention_bias = false;
  model.config.hidden_act = "silu";
  model.config.hidden_size = 2;
  model.config.intermediate_size = 2;
  model.config.max_position_embeddings = 8;
  model.config.model_type = "llama";
  model.config.num_attention_heads = 1;
  model.config.num_hidden_layers = 1;
  model.config.num_key_value_heads = 1;
  model.config.rms_norm_eps = 0.0;
  model.config.rope_theta = 10000.0;
  model.config.vocab_size = 3;

  model.weights.token_embedding =
      MakeFp32Tensor({3, 2}, {1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f});
  model.weights.final_norm = MakeFp32Tensor({2}, {1.0f, 1.0f});
  model.weights.lm_head =
      MakeFp32Tensor({3, 2}, {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f});

  model::LlamaHfLayerWeights layer;
  layer.input_layernorm = MakeFp32Tensor({2}, {1.0f, 1.0f});
  layer.post_attention_layernorm = MakeFp32Tensor({2}, {1.0f, 1.0f});
  layer.q_proj = Zeros({2, 2}, 4);
  layer.k_proj = Zeros({2, 2}, 4);
  layer.v_proj = Zeros({2, 2}, 4);
  layer.o_proj = Zeros({2, 2}, 4);
  layer.gate_proj = Zeros({2, 2}, 4);
  layer.up_proj = Zeros({2, 2}, 4);
  layer.down_proj = Zeros({2, 2}, 4);
  model.weights.layers.push_back(std::move(layer));
  return model;
}

}  // namespace

TEST(LlamaHfForwardTest, RunsOneTokenForward) {
  model::LlamaHfModel model = MakeTinyForwardModel();
  std::unique_ptr<model::LlamaBackend> backend =
      model::CreateLlamaBackend(base::DeviceType::kDeviceCPU);
  model::LlamaForwardState state = model::CreateLlamaForwardState(model.config);

  auto result = backend->ForwardToken(model, state, 0, 0);

  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->logits.size(), 3);
  EXPECT_EQ(result->next_token, 1);
  EXPECT_GT(result->logits.at<float>(1), result->logits.at<float>(0));
  EXPECT_GT(result->logits.at<float>(1), result->logits.at<float>(2));
}

TEST(LlamaHfForwardTest, BackendStateKeepsKvCacheAcrossTokens) {
  model::LlamaHfModel model = MakeTinyForwardModel();
  std::unique_ptr<model::LlamaBackend> backend =
      model::CreateLlamaBackend(base::DeviceType::kDeviceCPU);
  model::LlamaForwardState state = model::CreateLlamaForwardState(model.config);

  auto first = backend->ForwardToken(model, state, 0, 0);
  ASSERT_TRUE(first.ok()) << first.status();

  auto second = backend->ForwardToken(model, state, 1, 1);
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(second->next_token, 2);
}
