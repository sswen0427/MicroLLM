#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstring>

#include "base/base.h"
#include "tensor/tensor.h"

TEST(TensorCudaTest, TOCPU) {
  tensor::Tensor t1_cu = tensor::Tensor::allocate(
      base::DataType::kDataTypeInt32, {32, 32}, base::DeviceType::kDeviceCUDA);
  CHECK(!t1_cu.is_empty());

  int array[32 * 32];
  for (int i = 0; i < 32 * 32; i++) {
    array[i] = i;
  }
  cudaMemcpy(t1_cu.data<int>(), array, sizeof(array), cudaMemcpyHostToDevice);
  t1_cu.to_cpu();

  CHECK(t1_cu.device_type() == base::DeviceType::kDeviceCPU);

  int* cpu_ptr = t1_cu.data<int>();
  for (int i = 0; i < 32 * 32; ++i) {
    CHECK_EQ(*(cpu_ptr + i), i);
  }
}

TEST(TensorCudaTest, TOCUDA) {
  tensor::Tensor t1_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeInt32, {32, 32}, base::DeviceType::kDeviceCPU);
  CHECK(!t1_cpu.is_empty());

  int array[32 * 32];
  for (int i = 0; i < 32 * 32; i++) {
    array[i] = i;
  }
  std::memcpy(t1_cpu.data<int>(), array, sizeof(array));
  t1_cpu.to_cuda();

  int expected_array[32 * 32];
  cudaMemcpy(expected_array, t1_cpu.data<int>(), 32 * 32 * 4,
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    CHECK_EQ(expected_array[i], i);
  }
}

TEST(TensorCudaTest, CloneCUDA) {
  tensor::Tensor t1_cu = tensor::Tensor::allocate(
      base::DataType::kDataTypeInt32, {32, 32}, base::DeviceType::kDeviceCUDA);
  CHECK(!t1_cu.is_empty());
  int array[32 * 32];
  for (int i = 0; i < 32 * 32; i++) {
    array[i] = i;
  }
  cudaMemcpy(t1_cu.data<int>(), array, 32 * 32 * 4, cudaMemcpyHostToDevice);

  int expected_array[32 * 32];
  tensor::Tensor t2_cu = t1_cu.clone();
  cudaMemcpy(expected_array, t2_cu.data<int>(), sizeof(int) * 32 * 32,
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    EXPECT_EQ(expected_array[i], i);
  }
  EXPECT_EQ(t2_cu.data_type(), base::DataType::kDataTypeInt32);
  EXPECT_EQ(t2_cu.size(), 32 * 32);

  t2_cu.to_cpu();
  int* cpu_ptr = t2_cu.data<int>();
  for (int i = 0; i < 32 * 32; ++i) {
    CHECK_EQ(*(cpu_ptr + i), i);
  }
}
