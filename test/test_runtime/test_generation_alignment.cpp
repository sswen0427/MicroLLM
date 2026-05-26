#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "model/llama_hf_model_loader.h"
#include "runtime/generator.h"
#include "tokenizer/tokenizer.h"

namespace {

std::filesystem::path AlignmentModelDir() {
  const char* env_model_dir = std::getenv("MICRO_LLM_ALIGNMENT_MODEL_DIR");
  if (env_model_dir != nullptr && std::string(env_model_dir).size() > 0) {
    return env_model_dir;
  }

  const std::filesystem::path root_path = ROOT_PATH;
  return root_path / "data/my_tinyllama/AI-ModelScope" /
         "TinyLlama-1___1B-Chat-v1___0";
}

}  // namespace

TEST(GenerationAlignmentTest, TinyLlamaGreedyGenerationMatchesHfReference) {
  const std::filesystem::path model_dir = AlignmentModelDir();
  if (!std::filesystem::exists(model_dir / "config.json") ||
      !std::filesystem::exists(model_dir / "model.safetensors") ||
      !std::filesystem::exists(model_dir / "tokenizer.model")) {
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
  config.trace_steps = true;
  config.trace_top_k = 5;

  auto result_or =
      runtime::GenerateText(**model_or, **tokenizer_or, "I am a", config);
  ASSERT_TRUE(result_or.ok()) << result_or.status();

  const std::vector<int32_t> expected_prompt_tokens = {1, 306, 626, 263};
  const std::vector<int32_t> expected_generated_tokens = {
      13524, 310, 278, 1510, 29889, 306, 505, 1063};

  EXPECT_EQ(result_or->prompt_tokens, expected_prompt_tokens);
  EXPECT_EQ(result_or->tokens, expected_generated_tokens);
  ASSERT_EQ(result_or->steps.size(), expected_generated_tokens.size());

  for (size_t i = 0; i < expected_generated_tokens.size(); ++i) {
    EXPECT_EQ(result_or->steps[i].next_token_id, expected_generated_tokens[i])
        << "Mismatch at generation step " << i;
  }
}
