#include "generation_alignment_test_util.h"

TEST(GenerationAlignmentCudaTest,
     TinyLlamaGreedyGenerationMatchesHfReferenceOnCuda) {
  test_runtime::RunTinyLlamaGreedyAlignment(base::DeviceType::kDeviceCUDA);
}
