#include <gtest/gtest.h>

#include "base/types.h"
#include "model/hf_config.h"
#include "model/llama_kv_cache.h"

TEST(LlamaKvCacheTest, AllocatesLayerCachesWithExpectedShape) {
  model::HfLlamaConfig config;
  config.num_hidden_layers = 2;
  config.max_position_embeddings = 16;

  model::LlamaKvCache cache =
      model::LlamaKvCache::Allocate(config, /*kv_dim=*/8,
                                    base::DeviceType::kDeviceCPU);

  EXPECT_FALSE(cache.empty());
  EXPECT_EQ(cache.layer_count(), 2);
  EXPECT_EQ(cache.max_seq_len(), 16);
  EXPECT_EQ(cache.kv_dim(), 8);
  EXPECT_EQ(cache.seq_len(), 0);
  EXPECT_EQ(cache.device_type(), base::DeviceType::kDeviceCPU);

  for (int32_t layer = 0; layer < cache.layer_count(); ++layer) {
    EXPECT_EQ(cache.key(layer).device_type(), base::DeviceType::kDeviceCPU);
    EXPECT_EQ(cache.value(layer).device_type(), base::DeviceType::kDeviceCPU);
    EXPECT_EQ(cache.key(layer).data_type(), base::DataType::kDataTypeFp32);
    EXPECT_EQ(cache.value(layer).data_type(), base::DataType::kDataTypeFp32);
    EXPECT_EQ(cache.key(layer).dims_size(), 2);
    EXPECT_EQ(cache.value(layer).dims_size(), 2);
    EXPECT_EQ(cache.key(layer).get_dim(0), 16);
    EXPECT_EQ(cache.key(layer).get_dim(1), 8);
    EXPECT_EQ(cache.value(layer).get_dim(0), 16);
    EXPECT_EQ(cache.value(layer).get_dim(1), 8);
  }
}

TEST(LlamaKvCacheTest, TracksWrittenSequenceLength) {
  model::HfLlamaConfig config;
  config.num_hidden_layers = 1;
  config.max_position_embeddings = 16;

  model::LlamaKvCache cache =
      model::LlamaKvCache::Allocate(config, /*kv_dim=*/8,
                                    base::DeviceType::kDeviceCPU);

  cache.CommitToken(0);
  EXPECT_EQ(cache.seq_len(), 1);

  cache.CommitToken(1);
  EXPECT_EQ(cache.seq_len(), 2);

  cache.CommitToken(2);
  EXPECT_EQ(cache.seq_len(), 3);

  cache.Reset();
  EXPECT_EQ(cache.seq_len(), 0);

  cache.CommitToken(0);
  EXPECT_EQ(cache.seq_len(), 1);
}

TEST(LlamaKvCacheTest, TracksPrefillCommit) {
  model::HfLlamaConfig config;
  config.num_hidden_layers = 1;
  config.max_position_embeddings = 16;

  model::LlamaKvCache cache =
      model::LlamaKvCache::Allocate(config, /*kv_dim=*/8,
                                    base::DeviceType::kDeviceCPU);

  cache.CommitTokens(/*start_position=*/0, /*token_count=*/4);
  EXPECT_EQ(cache.seq_len(), 4);

  cache.CommitTokens(/*start_position=*/4, /*token_count=*/3);
  EXPECT_EQ(cache.seq_len(), 7);
}
