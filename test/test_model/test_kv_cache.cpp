#include <gtest/gtest.h>

#include "base/types.h"
#include "model/kv_cache.h"

TEST(KvCacheTest, AllocatesLayerCachesWithExpectedShape) {
  model::KvCache cache =
      model::KvCache::Allocate(/*layer_count=*/2, /*max_seq_len=*/16,
                               /*kv_dim=*/8, base::DeviceType::kDeviceCPU);

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

TEST(KvCacheTest, TracksWrittenSequenceLength) {
  model::KvCache cache =
      model::KvCache::Allocate(/*layer_count=*/1, /*max_seq_len=*/16,
                               /*kv_dim=*/8, base::DeviceType::kDeviceCPU);

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

TEST(KvCacheTest, TracksPrefillCommit) {
  model::KvCache cache =
      model::KvCache::Allocate(/*layer_count=*/1, /*max_seq_len=*/16,
                               /*kv_dim=*/8, base::DeviceType::kDeviceCPU);

  cache.CommitTokens(/*start_position=*/0, /*token_count=*/4);
  EXPECT_EQ(cache.seq_len(), 4);

  cache.CommitTokens(/*start_position=*/4, /*token_count=*/3);
  EXPECT_EQ(cache.seq_len(), 7);
}
