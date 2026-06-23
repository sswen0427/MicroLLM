#include "generation_alignment_test_util.h"

TEST(GenerationAlignmentTest, TinyLlamaGreedyGenerationMatchesHfReference) {
  test_runtime::RunTinyLlamaGreedyAlignment(base::DeviceType::kDeviceCPU);
}
