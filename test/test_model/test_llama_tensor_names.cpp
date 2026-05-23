#include "model/llama_tensor_names.h"

#include <gtest/gtest.h>

namespace {

TEST(LlamaTensorNamesTest, BuildsGlobalTensorNames) {
  EXPECT_EQ(model::LlamaTensorName(model::LlamaTensorKind::kTokenEmbedding),
            "model.embed_tokens.weight");
  EXPECT_EQ(model::LlamaTensorName(model::LlamaTensorKind::kFinalNorm),
            "model.norm.weight");
  EXPECT_EQ(model::LlamaTensorName(model::LlamaTensorKind::kLmHead),
            "lm_head.weight");
}

TEST(LlamaTensorNamesTest, BuildsLayerTensorNames) {
  EXPECT_EQ(model::LlamaLayerTensorName(10, model::LlamaTensorKind::kQProj),
            "model.layers.10.self_attn.q_proj.weight");
  EXPECT_EQ(model::LlamaLayerTensorName(10, model::LlamaTensorKind::kGateProj),
            "model.layers.10.mlp.gate_proj.weight");
}

}  // namespace
