#include <gtest/gtest.h>

#include <filesystem>

#include "base/alias.h"
#include "op/encode.h"

TEST(SpeEncodeTest, SpeEncode) {
  Path root_path = ROOT_PATH;
  Path path = root_path / "data/my_tinyllama/AI-ModelScope" /
              "TinyLlama-1___1B-Chat-v1___0/tokenizer.model";
  op::SpeEncodeLayer tokenizer(path, true, true);
  EXPECT_EQ(tokenizer.vocab_size(), 32000);

  std::string input_text = "Hello world!";
  std::vector<int32_t> encoded_ids = tokenizer.encode(input_text);
  std::vector<int> expected_encoded_ids = {1, 15043, 3186, 29991, 2};
  EXPECT_EQ(encoded_ids, expected_encoded_ids);

  std::string decoded_text = tokenizer.decode(encoded_ids);
  EXPECT_EQ(decoded_text, input_text);
  EXPECT_EQ("", tokenizer.decode(encoded_ids[0]));
  EXPECT_EQ("Hello", tokenizer.decode(encoded_ids[1]));
  EXPECT_EQ("world", tokenizer.decode(encoded_ids[2]));
  EXPECT_EQ("!", tokenizer.decode(encoded_ids[3]));
  EXPECT_EQ("", tokenizer.decode(encoded_ids[4]));
  EXPECT_TRUE(tokenizer.is_sentence_ending(encoded_ids[4]));
  EXPECT_FALSE(tokenizer.is_sentence_ending(encoded_ids[3]));
}