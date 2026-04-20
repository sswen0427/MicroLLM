#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "base/buffer.h"
#include "op/kernels/kernels_interface.h"
using namespace kernel;
TEST(test_matmul_cu, matmul_linear_stream5) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {4}, alloc_cpu);
  tensor::Tensor weight = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {4, 4}, alloc_cpu);

  for (int i = 0; i < 4; ++i) {
    input.at<float>(i) = float(i);
  }

  for (int i = 0; i < 16; ++i) {
    weight.at<float>(i) = float(i);
  }
  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor out_cu =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {4}, alloc_cu);
  tensor::Tensor out_cpu =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {4}, alloc_cpu);

  auto* config = new base::CudaConfig;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  config->stream = stream;
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input, weight,
                                                           out_cu, 1.f, config);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu, weight_cpu,
                                                          out_cpu, 1.f, config);

  out_cu.to_cpu();
  for (int i = 0; i < out_cu.size(); ++i) {
    ASSERT_EQ(out_cu.at<float>(i), out_cpu.at<float>(i));
  }
}

TEST(test_matmul_cu, matmul_linear_course) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {3}, alloc_cpu);
  tensor::Tensor weight = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {3, 3}, alloc_cpu);

  input.at<float>(0) = float(1);
  input.at<float>(1) = float(1);
  input.at<float>(2) = float(-1);

  for (int i = 1; i <= 9; ++i) {
    weight.at<float>(i - 1) = float(i);
  }
  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor out_cpu =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {3}, alloc_cpu);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(
      input_cpu, weight_cpu, out_cpu, 1.f, nullptr);

  ASSERT_EQ(out_cpu.at<float>(0), 0);
  ASSERT_EQ(out_cpu.at<float>(1), 3);
  ASSERT_EQ(out_cpu.at<float>(2), 6);
}

TEST(test_matmul_cu, matmul_linear_course_cuda) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {3}, alloc_cpu);
  tensor::Tensor weight = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {3, 3}, alloc_cpu);

  input.at<float>(0) = float(1);
  input.at<float>(1) = float(1);
  input.at<float>(2) = float(-1);

  for (int i = 1; i <= 9; ++i) {
    weight.at<float>(i - 1) = float(i);
  }

  input.to_cuda();
  weight.to_cuda();

  tensor::Tensor out_cu =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {3}, alloc_cu);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(
      input, weight, out_cu, 1.f, nullptr);

  tensor::Tensor out_cpu = out_cu.clone();
  out_cpu.to_cpu();

  ASSERT_EQ(out_cpu.at<float>(0), 0);
  ASSERT_EQ(out_cpu.at<float>(1), 3);
  ASSERT_EQ(out_cpu.at<float>(2), 6);
}