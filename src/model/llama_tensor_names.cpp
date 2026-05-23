#include "model/llama_tensor_names.h"

#include <absl/strings/str_cat.h>
#include <glog/logging.h>

namespace model {
namespace {

std::string LlamaLayerTensorSuffix(LlamaTensorKind kind) {
  switch (kind) {
    case LlamaTensorKind::kInputLayerNorm:
      return "input_layernorm.weight";
    case LlamaTensorKind::kPostAttentionLayerNorm:
      return "post_attention_layernorm.weight";
    case LlamaTensorKind::kQProj:
      return "self_attn.q_proj.weight";
    case LlamaTensorKind::kKProj:
      return "self_attn.k_proj.weight";
    case LlamaTensorKind::kVProj:
      return "self_attn.v_proj.weight";
    case LlamaTensorKind::kOProj:
      return "self_attn.o_proj.weight";
    case LlamaTensorKind::kGateProj:
      return "mlp.gate_proj.weight";
    case LlamaTensorKind::kUpProj:
      return "mlp.up_proj.weight";
    case LlamaTensorKind::kDownProj:
      return "mlp.down_proj.weight";
    default:
      LOG(FATAL) << "Not a per-layer LLaMA tensor kind";
      return "";
  }
}

}  // namespace

std::string LlamaTensorName(LlamaTensorKind kind) {
  switch (kind) {
    case LlamaTensorKind::kTokenEmbedding:
      return "model.embed_tokens.weight";
    case LlamaTensorKind::kFinalNorm:
      return "model.norm.weight";
    case LlamaTensorKind::kLmHead:
      return "lm_head.weight";
    default:
      LOG(FATAL) << "Not a global LLaMA tensor kind";
      return "";
  }
}

std::string LlamaLayerTensorName(int32_t layer, LlamaTensorKind kind) {
  CHECK_GE(layer, 0);
  return absl::StrCat("model.layers.", layer, ".",
                      LlamaLayerTensorSuffix(kind));
}

}  // namespace model
