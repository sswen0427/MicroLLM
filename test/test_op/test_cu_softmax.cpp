#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <random>

#include "base/buffer.h"
#include "op/kernels/kernels_interface.h"

TEST(SoftmaxTest, OneDimensionCPU) {
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1 =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {3}, alloc_cpu);
  t1.at<float>(0) = 1.0;
  t1.at<float>(1) = 2.0;
  t1.at<float>(2) = 3.0;

  kernel::get_softmax_kernel(base::DeviceType::kDeviceCPU)(t1, nullptr);
  EXPECT_NEAR(t1.at<float>(0), 0.09003, 1e-5);
  EXPECT_NEAR(t1.at<float>(1), 0.24473, 1e-5);
  EXPECT_NEAR(t1.at<float>(2), 0.66524, 1e-5);
}

TEST(SoftmaxTest, DISABLED_Nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor in_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {size}, alloc_cpu);

  srand(0);
  for (int i = 0; i < size; ++i) {
    in_cpu.at<float>(i) = rand() % 31;
  }

  tensor::Tensor in_cu = in_cpu.clone();
  in_cu.to_cuda();

  kernel::get_softmax_kernel(base::DeviceType::kDeviceCUDA)(in_cu, nullptr);
  kernel::get_softmax_kernel(base::DeviceType::kDeviceCPU)(in_cpu, nullptr);

  in_cu.to_cpu();

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(in_cpu.at<float>(i), in_cu.at<float>(i), 1e-5f);
  }
}

TEST(SoftmaxTest, DISABLED_Stream1) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 72 * 151;

  tensor::Tensor in_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {size}, alloc_cpu);

  srand(0);
  for (int i = 0; i < size; ++i) {
    in_cpu.at<float>(i) = rand() % 31;
  }

  tensor::Tensor in_cu = in_cpu.clone();
  in_cu.to_cuda();

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_softmax_kernel(base::DeviceType::kDeviceCUDA)(in_cu, stream);
  kernel::get_softmax_kernel(base::DeviceType::kDeviceCPU)(in_cpu, nullptr);

  in_cu.to_cpu();

  for (int i = 0; i < size; ++i) {
    EXPECT_NEAR(in_cpu.at<float>(i), in_cu.at<float>(i), 1e-5f);
  }
}

TEST(SoftmaxTest, DISABLED_Stream2) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 72 * 18;

  tensor::Tensor in_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {size}, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.at<float>(i) = dist(mt);
  }

  tensor::Tensor in_cu = in_cpu.clone();
  in_cu.to_cuda();

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_softmax_kernel(base::DeviceType::kDeviceCUDA)(in_cu, stream);
  kernel::get_softmax_kernel(base::DeviceType::kDeviceCPU)(in_cpu, nullptr);
  in_cu.to_cpu();

  for (int i = 0; i < size; ++i) {
    EXPECT_NEAR(in_cpu.at<float>(i), in_cu.at<float>(i), 1e-5f);
  }
}

TEST(SoftmaxTest, DISABLED_Stream3) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 1;

  tensor::Tensor in_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {size}, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.at<float>(i) = dist(mt);
  }

  tensor::Tensor in_cu = in_cpu.clone();
  in_cu.to_cuda();

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_softmax_kernel(base::DeviceType::kDeviceCUDA)(in_cu, stream);
  kernel::get_softmax_kernel(base::DeviceType::kDeviceCPU)(in_cpu, nullptr);
  in_cu.to_cpu();

  for (int i = 0; i < size; ++i) {
    EXPECT_NEAR(in_cpu.at<float>(i), in_cu.at<float>(i), 1e-5f);
  }
}