#pragma once

#include <cstdint>
#include <string>

namespace model {

enum class LlamaTensorKind {
  kTokenEmbedding,
  kFinalNorm,
  kLmHead,
  kInputLayerNorm,
  kPostAttentionLayerNorm,
  kQProj,
  kKProj,
  kVProj,
  kOProj,
  kGateProj,
  kUpProj,
  kDownProj,
};

std::string LlamaTensorName(LlamaTensorKind kind);

std::string LlamaLayerTensorName(int32_t layer, LlamaTensorKind kind);

}  // namespace model
