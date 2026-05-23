#include "tensor/tensor_view.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

TEST(TensorViewTest, ValidatesBorrowedTensorMetadata) {
  const uint16_t data[4] = {};

  const tensor::TensorView view{
      .data_type = base::DataType::kDataTypeBf16,
      .device_type = base::DeviceType::kDeviceCPU,
      .shape = {2, 2},
      .data = data,
      .byte_size = sizeof(data),
  };

  EXPECT_TRUE(view.is_valid());
  EXPECT_EQ(view.element_count(), 4);
  EXPECT_EQ(view.expected_byte_size(), sizeof(data));
  EXPECT_EQ(base::DataTypeSize(view.data_type), 2);
}

TEST(TensorViewTest, RejectsEmptyView) {
  const tensor::TensorView view;

  EXPECT_FALSE(view.is_valid());
  EXPECT_EQ(view.element_count(), 0);
}

TEST(TensorViewTest, RejectsByteSizeMismatch) {
  const uint16_t data[4] = {};

  const tensor::TensorView view{
      .data_type = base::DataType::kDataTypeBf16,
      .device_type = base::DeviceType::kDeviceCPU,
      .shape = {2, 2},
      .data = data,
      .byte_size = sizeof(data) - 1,
  };

  EXPECT_FALSE(view.is_valid());
  EXPECT_EQ(view.expected_byte_size(), sizeof(data));
}

}  // namespace
