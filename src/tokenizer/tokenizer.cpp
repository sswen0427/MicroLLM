#include "tokenizer/tokenizer.h"

#include <absl/status/status.h>
#include <absl/strings/str_cat.h>
#include <sentencepiece_processor.h>

#include <utility>

namespace tokenizer {

SentencePieceTokenizer::SentencePieceTokenizer(TokenizerOptions options)
    : options_(options),
      processor_(std::make_unique<sentencepiece::SentencePieceProcessor>()) {}

SentencePieceTokenizer::~SentencePieceTokenizer() = default;

absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>>
SentencePieceTokenizer::Load(const std::string& model_path,
                             TokenizerOptions options) {
  auto tokenizer = std::unique_ptr<SentencePieceTokenizer>(
      new SentencePieceTokenizer(options));
  const auto status = tokenizer->processor_->Load(model_path);
  if (!status.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to load SentencePiece model: ", model_path,
                     ", error: ", status.ToString()));
  }
  return tokenizer;
}

std::vector<int32_t> SentencePieceTokenizer::Encode(
    const std::string& text) const {
  std::vector<int32_t> token_ids = processor_->EncodeAsIds(text);
  if (options_.add_bos) {
    token_ids.insert(token_ids.begin(), processor_->bos_id());
  }
  if (options_.add_eos) {
    token_ids.push_back(processor_->eos_id());
  }
  return token_ids;
}

std::string SentencePieceTokenizer::Decode(int32_t token_id) const {
  return Decode(std::vector<int32_t>{token_id});
}

std::string SentencePieceTokenizer::Decode(
    const std::vector<int32_t>& token_ids) const {
  return processor_->DecodeIds(token_ids);
}

bool SentencePieceTokenizer::IsEndToken(int32_t token_id) const {
  return token_id == processor_->eos_id();
}

int32_t SentencePieceTokenizer::VocabSize() const {
  return processor_->GetPieceSize();
}

}  // namespace tokenizer
