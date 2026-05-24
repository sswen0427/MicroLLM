#pragma once

#include <absl/status/statusor.h>

#include <cstdint>
#include <memory>
#include <sentencepiece_processor.h>
#include <string>
#include <vector>

namespace tokenizer {

struct TokenizerOptions {
  bool add_bos = true;
  bool add_eos = false;
};

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  virtual std::vector<int32_t> Encode(const std::string& text) const = 0;

  virtual std::string Decode(const std::vector<int32_t>& token_ids) const = 0;

  virtual bool IsEndToken(int32_t token_id) const = 0;

  virtual int32_t VocabSize() const = 0;
};

class SentencePieceTokenizer final : public Tokenizer {
 public:
  static absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>> Load(
      const std::string& model_path, TokenizerOptions options);

  ~SentencePieceTokenizer() override;

  std::vector<int32_t> Encode(const std::string& text) const override;

  std::string Decode(const std::vector<int32_t>& token_ids) const override;

  bool IsEndToken(int32_t token_id) const override;

  int32_t VocabSize() const override;

 private:
  explicit SentencePieceTokenizer(TokenizerOptions options);

  TokenizerOptions options_;
  std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;
};

}  // namespace tokenizer
