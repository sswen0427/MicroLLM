#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tensor/tensor.h>

#include "base/base.h"
#include "base/buffer.h"

TEST(TensorTest, TOCPU) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cu = tensor::Tensor::allocate(
      base::DataType::kDataTypeInt32, {32, 32}, alloc_cu);
  CHECK(!t1_cu.is_empty());

  int array[32 * 32];
  for (int i = 0; i < 32 * 32; i++) {
    array[i] = i;
  }
  base::Buffer buffer_cpu(32 * 32 * 4, nullptr, array);
  t1_cu.get_buffer().get()->copy_from(buffer_cpu);
  t1_cu.to_cpu();

  CHECK(t1_cu.device_type() == base::DeviceType::kDeviceCPU);

  int* cpu_ptr = t1_cu.ptr<int>();
  for (int i = 0; i < 32 * 32; ++i) {
    CHECK_EQ(*(cpu_ptr + i), i);
  }
}

TEST(TensorTest, TOCUDA) {
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeInt32, {32, 32}, alloc_cpu);
  CHECK(!t1_cpu.is_empty());

  int array[32 * 32];
  for (int i = 0; i < 32 * 32; i++) {
    array[i] = i;
  }
  base::Buffer buffer_cpu(32 * 32 * 4, nullptr, array);
  t1_cpu.get_buffer().get()->copy_from(buffer_cpu);
  t1_cpu.to_cuda();

  int expected_array[32 * 32];
  cudaMemcpy(expected_array, t1_cpu.ptr<int>(), 32 * 32 * 4,
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    CHECK_EQ(expected_array[i], i);
  }
}

TEST(TensorTest, CloneCUDA) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cu = tensor::Tensor::allocate(
      base::DataType::kDataTypeInt32, {32, 32}, alloc_cu);
  CHECK(!t1_cu.is_empty());
  int array[32 * 32];
  for (int i = 0; i < 32 * 32; i++) {
    array[i] = i;
  }
  cudaMemcpy(t1_cu.ptr<int>(), array, 32 * 32 * 4, cudaMemcpyHostToDevice);

  int expected_array[32 * 32];
  tensor::Tensor t2_cu = t1_cu.clone();
  cudaMemcpy(expected_array, t2_cu.ptr<float>(), sizeof(float) * 32 * 32,
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    EXPECT_EQ(expected_array[i], i);
  }
  EXPECT_EQ(t2_cu.data_type(), base::DataType::kDataTypeInt32);
  EXPECT_EQ(t2_cu.size(), 32 * 32);

  t2_cu.to_cpu();
  int* cpu_ptr = t2_cu.ptr<int>();
  for (int i = 0; i < 32 * 32; ++i) {
    CHECK_EQ(*(cpu_ptr + i), i);
  }
}

TEST(TensorTest, CloneCPU) {
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeInt32, {32, 32}, alloc_cpu);
  CHECK(!t1_cpu.is_empty());
  for (int i = 0; i < 32 * 32; ++i) {
    t1_cpu.at<int>(i) = i;
  }

  tensor::Tensor t2_cpu = t1_cpu.clone();
  int expected_array[32 * 32];
  std::memcpy(expected_array, t2_cpu.ptr<int>(), 32 * 32 * 4);
  for (int i = 0; i < 32 * 32; ++i) {
    EXPECT_EQ(expected_array[i], i);
  }
  EXPECT_EQ(t2_cpu.data_type(), base::DataType::kDataTypeInt32);
  EXPECT_EQ(t2_cpu.size(), 32 * 32);
}
