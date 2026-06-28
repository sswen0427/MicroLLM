#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <set>
#include <vector>

#include "model/llama_hf_model_loader.h"
#include "runtime/generator.h"
#include "tokenizer/tokenizer.h"

namespace test_runtime {

inline constexpr float kTop1LogitAbsTolerance = 7e-2f;

inline std::filesystem::path TinyLlamaAlignmentModelDir() {
  const std::filesystem::path root_path = ROOT_PATH;
  return root_path / "data/my_tinyllama/AI-ModelScope" /
         "TinyLlama-1___1B-Chat-v1___0";
}

inline bool TinyLlamaAlignmentModelExists(
    const std::filesystem::path &model_dir) {
  return std::filesystem::exists(model_dir / "config.json") &&
         std::filesystem::exists(model_dir / "model.safetensors") &&
         std::filesystem::exists(model_dir / "tokenizer.model");
}

inline void ExpectTinyLlamaReferenceMatch(
    const runtime::GenerationResult &result) {
  const std::vector<int32_t> expected_prompt_tokens = {1, 306, 626, 263};
  const std::vector<int32_t> expected_generated_tokens = {
      13524, 310, 278, 1510, 29889, 306, 505, 1063};
  const std::vector<runtime::GenerationStep> expected_steps = {
      {.step = 0,
       .position = 3,
       .input_token_id = 263,
       .next_token_id = 13524,
       .top_logits = {{13524, 8.0625f},
                      {2217, 8.0f},
                      {716, 7.71875f},
                      {4802, 7.65625f},
                      {4509, 7.65625f}}},
      {.step = 1,
       .position = 4,
       .input_token_id = 13524,
       .next_token_id = 310,
       .top_logits = {{310, 14.9375f},
                      {29889, 12.75f},
                      {29991, 12.0f},
                      {29892, 11.5f},
                      {322, 10.9375f}}},
      {.step = 2,
       .position = 5,
       .input_token_id = 310,
       .next_token_id = 278,
       .top_logits = {{278, 9.1875f},
                      {596, 9.0625f},
                      {1716, 8.5625f},
                      {445, 8.0625f},
                      {599, 7.90625f}}},
      {.step = 3,
       .position = 6,
       .input_token_id = 278,
       .next_token_id = 1510,
       .top_logits = {{1510, 7.65625f},
                      {3143, 7.15625f},
                      {4148, 7.125f},
                      {716, 7.0f},
                      {22037, 6.9375f}}},
      {.step = 4,
       .position = 7,
       .input_token_id = 1510,
       .next_token_id = 29889,
       .top_logits = {{29889, 11.3125f},
                      {322, 11.0625f},
                      {29892, 10.8125f},
                      {29991, 9.875f},
                      {29973, 9.5f}}},
      {.step = 5,
       .position = 8,
       .input_token_id = 29889,
       .next_token_id = 306,
       .top_logits = {{306, 12.9375f},
                      {1815, 12.0f},
                      {2, 11.5f},
                      {739, 11.5f},
                      {13, 11.3125f}}},
      {.step = 6,
       .position = 9,
       .input_token_id = 306,
       .next_token_id = 505,
       .top_logits = {{505, 13.25f},
                      {5360, 12.9375f},
                      {626, 12.75f},
                      {1348, 12.375f},
                      {29915, 12.3125f}}},
      {.step = 7,
       .position = 10,
       .input_token_id = 505,
       .next_token_id = 1063,
       .top_logits = {{1063, 13.0f},
                      {20654, 12.5625f},
                      {3595, 12.0f},
                      {2337, 11.4375f},
                      {263, 11.3125f}}},
  };

  EXPECT_EQ(result.prompt_tokens, expected_prompt_tokens);
  EXPECT_EQ(result.tokens, expected_generated_tokens);
  EXPECT_EQ(result.text, "fan of the show. I have been");
  ASSERT_EQ(result.steps.size(), expected_steps.size());

  for (size_t i = 0; i < expected_steps.size(); ++i) {
    const runtime::GenerationStep &actual_step = result.steps[i];
    const runtime::GenerationStep &expected_step = expected_steps[i];
    EXPECT_EQ(actual_step.step, expected_step.step)
        << "Mismatch at generation step " << i;
    EXPECT_EQ(actual_step.position, expected_step.position)
        << "Mismatch at generation step " << i;
    EXPECT_EQ(actual_step.input_token_id, expected_step.input_token_id)
        << "Mismatch at generation step " << i;
    EXPECT_EQ(actual_step.next_token_id, expected_step.next_token_id)
        << "Mismatch at generation step " << i;
    ASSERT_EQ(actual_step.top_logits.size(), expected_step.top_logits.size())
        << "Mismatch at generation step " << i;

    EXPECT_EQ(actual_step.top_logits.front().first,
              expected_step.top_logits.front().first)
        << "Mismatch at generation step " << i;
    EXPECT_NEAR(actual_step.top_logits.front().second,
                expected_step.top_logits.front().second, kTop1LogitAbsTolerance)
        << "Mismatch at generation step " << i;

    std::set<int32_t> actual_top_token_ids;
    std::set<int32_t> expected_top_token_ids;
    for (const auto &[token_id, logit] : actual_step.top_logits) {
      actual_top_token_ids.insert(token_id);
    }
    for (const auto &[token_id, logit] : expected_step.top_logits) {
      expected_top_token_ids.insert(token_id);
    }
    EXPECT_EQ(actual_top_token_ids, expected_top_token_ids)
        << "Mismatch at generation step " << i;
  }
}

inline void RunTinyLlamaGreedyAlignment(base::DeviceType device_type) {
  const std::filesystem::path model_dir = TinyLlamaAlignmentModelDir();
  if (!TinyLlamaAlignmentModelExists(model_dir)) {
    GTEST_SKIP() << "Alignment model is not available: " << model_dir;
  }

  auto model_or = model::LoadLlamaHfModel(model_dir.string());
  ASSERT_TRUE(model_or.ok()) << model_or.status();

  auto tokenizer_or = tokenizer::SentencePieceTokenizer::Load(
      (model_dir / "tokenizer.model").string(),
      tokenizer::TokenizerOptions{.add_bos = true, .add_eos = false});
  ASSERT_TRUE(tokenizer_or.ok()) << tokenizer_or.status();

  runtime::GenerationConfig config;
  config.max_new_tokens = 8;
  config.device_type = device_type;
  config.trace_steps = true;
  config.trace_top_k = 5;

  auto result_or =
      runtime::GenerateText(**model_or, **tokenizer_or, "I am a", config);
  ASSERT_TRUE(result_or.ok()) << result_or.status();
  ExpectTinyLlamaReferenceMatch(*result_or);
}

}  // namespace test_runtime
