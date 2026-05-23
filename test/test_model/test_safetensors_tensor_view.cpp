#include "model/safetensors_tensor_view.h"

#include <absl/status/status.h>
#include <gtest/gtest.h>
#include <safetensors.hh>

#include <cstdint>
#include <vector>

namespace {

TEST(SafetensorsTensorViewTest, BuildsViewFromStorageTensor) {
  safetensors::safetensors_t safetensors;
  safetensors.storage.resize(8);

  safetensors::tensor_t tensor;
  tensor.dtype = safetensors::dtype::kBFLOAT16;
  tensor.shape = {2, 2};
  tensor.data_offsets = {0, 8};
  safetensors.tensors.insert("weight", tensor);

  auto view_or = model::GetSafetensorsTensorView(safetensors, "weight");

  ASSERT_TRUE(view_or.ok()) << view_or.status().ToString();
  EXPECT_EQ(view_or->data_type, base::DataType::kDataTypeBf16);
  EXPECT_EQ(view_or->device_type, base::DeviceType::kDeviceCPU);
  EXPECT_EQ(view_or->shape, (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(view_or->data, safetensors.storage.data());
  EXPECT_EQ(view_or->byte_size, 8);
}

TEST(SafetensorsTensorViewTest, ReturnsNotFoundForMissingTensor) {
  const safetensors::safetensors_t safetensors;

  auto view_or = model::GetSafetensorsTensorView(safetensors, "missing");

  EXPECT_FALSE(view_or.ok());
  EXPECT_EQ(view_or.status().code(), absl::StatusCode::kNotFound);
}

}  // namespace
