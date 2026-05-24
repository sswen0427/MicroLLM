#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "tokenizer/tokenizer.h"

TEST(SentencePieceTokenizerTest, EncodesAndDecodesText) {
  using Path = std::filesystem::path;
  Path root_path = ROOT_PATH;
  Path path = root_path / "data/my_tinyllama/AI-ModelScope" /
              "TinyLlama-1___1B-Chat-v1___0/tokenizer.model";
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "Tokenizer model is not available: " << path;
  }

  auto tokenizer_or = tokenizer::SentencePieceTokenizer::Load(
      path.string(), tokenizer::TokenizerOptions{.add_bos = true,
                                                 .add_eos = true});
  ASSERT_TRUE(tokenizer_or.ok()) << tokenizer_or.status();
  const auto& tokenizer = **tokenizer_or;
  EXPECT_EQ(tokenizer.VocabSize(), 32000);

  const std::string input_text = "Hello world!";
  const std::vector<int32_t> encoded_ids = tokenizer.Encode(input_text);
  const std::vector<int32_t> expected_encoded_ids = {1, 15043, 3186, 29991, 2};
  EXPECT_EQ(encoded_ids, expected_encoded_ids);

  EXPECT_EQ(tokenizer.Decode(encoded_ids), input_text);
  EXPECT_EQ(tokenizer.Decode(encoded_ids[0]), "");
  EXPECT_EQ(tokenizer.Decode(encoded_ids[1]), "Hello");
  EXPECT_EQ(tokenizer.Decode(encoded_ids[2]), "world");
  EXPECT_EQ(tokenizer.Decode(encoded_ids[3]), "!");
  EXPECT_EQ(tokenizer.Decode(encoded_ids[4]), "");
  EXPECT_TRUE(tokenizer.IsEndToken(encoded_ids[4]));
  EXPECT_FALSE(tokenizer.IsEndToken(encoded_ids[3]));
}
