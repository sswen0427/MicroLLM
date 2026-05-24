#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstring>

#include "base/types.h"
#include "tensor/tensor.h"

TEST(TensorTest, CloneCPU) {
  tensor::Tensor t1_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeInt32, {32, 32}, base::DeviceType::kDeviceCPU);
  CHECK(!t1_cpu.is_empty());
  for (int i = 0; i < 32 * 32; ++i) {
    t1_cpu.at<int>(i) = i;
  }

  tensor::Tensor t2_cpu = t1_cpu.clone();
  int expected_array[32 * 32];
  std::memcpy(expected_array, t2_cpu.data<int>(), 32 * 32 * 4);
  for (int i = 0; i < 32 * 32; ++i) {
    EXPECT_EQ(expected_array[i], i);
  }
  EXPECT_EQ(t2_cpu.data_type(), base::DataType::kDataTypeInt32);
  EXPECT_EQ(t2_cpu.size(), 32 * 32);
}

TEST(TensorTest, FromExternalCPUTracksDeviceType) {
  int array[4] = {1, 2, 3, 4};

  tensor::Tensor tensor = tensor::Tensor::from_external_cpu(
      base::DataType::kDataTypeInt32, {2, 2}, array);

  EXPECT_EQ(tensor.device_type(), base::DeviceType::kDeviceCPU);
  EXPECT_EQ(tensor.size(), 4);
  EXPECT_EQ(tensor.byte_size(), sizeof(array));
  EXPECT_EQ(tensor.data<int>(), array);
  EXPECT_TRUE(tensor.is_external());
  EXPECT_FALSE(tensor.owns_memory());
}

TEST(TensorTest, CloneExternalCPUTensorOwnsCopiedMemory) {
  int array[4] = {1, 2, 3, 4};
  tensor::Tensor external = tensor::Tensor::from_external_cpu(
      base::DataType::kDataTypeInt32, {4}, array);

  tensor::Tensor cloned = external.clone();

  EXPECT_EQ(cloned.device_type(), base::DeviceType::kDeviceCPU);
  EXPECT_EQ(cloned.size(), external.size());
  EXPECT_NE(cloned.data<int>(), array);
  EXPECT_FALSE(cloned.is_external());
  EXPECT_TRUE(cloned.owns_memory());
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(cloned.at<int>(i), array[i]);
  }
}

TEST(TensorTest, RejectsInvalidShape) {
  EXPECT_DEATH((void)tensor::Tensor::allocate(base::DataType::kDataTypeInt32,
                                              {}, base::DeviceType::kDeviceCPU),
               "Tensor dims must not be empty");
  EXPECT_DEATH(
      (void)tensor::Tensor::allocate(base::DataType::kDataTypeInt32, {2, 0},
                                     base::DeviceType::kDeviceCPU),
      "Tensor dim must be positive");
}

TEST(TensorTest, RejectsUnknownDataType) {
  int array[4] = {};

  EXPECT_DEATH(
      (void)tensor::Tensor::allocate(base::DataType::kDataTypeUnknown, {4},
                                     base::DeviceType::kDeviceCPU),
      "Tensor data type must be known");
  EXPECT_DEATH((void)tensor::Tensor::from_external_cpu(
                   base::DataType::kDataTypeUnknown, {4}, array),
               "Tensor data type must be known");
}

TEST(TensorTest, RejectsDataAccessOnEmptyTensor) {
  const tensor::Tensor tensor;
  EXPECT_DEATH((void)tensor.data<int32_t>(), "data area buffer");
}
