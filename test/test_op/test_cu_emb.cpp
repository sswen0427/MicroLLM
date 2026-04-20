#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "base/buffer.h"
#include "op/kernels/kernels_interface.h"
TEST(test_emb_cu, emb1_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t token = 4;
  int32_t dim = 512;
  int32_t size = 2048;

  tensor::Tensor input =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {1}, alloc_cpu);
  input.at<int32_t>(0) = 1;

  tensor::Tensor weight = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {token, dim}, alloc_cpu);
  tensor::Tensor output =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {dim}, alloc_cu);

  for (int i = 0; i < size; ++i) {
    weight.at<float>(i) = static_cast<float>(i);
  }
  weight.to_cuda();
  kernel::get_emb_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output,
                                                        token, nullptr);

  output.to_cpu();
  for (int i = 0; i < dim; ++i) {
    ASSERT_EQ(output.at<float>(i), 512 + i);
  }
}

TEST(test_emb_cu, emb2_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t token = 4;
  int32_t dim = 512;
  int32_t size = 2048;

  tensor::Tensor input =
      tensor::Tensor::allocate(base::DataType::kDataTypeInt32, {1}, alloc_cpu);
  input.at<int32_t>(0) = 2;

  tensor::Tensor weight = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {token, dim}, alloc_cpu);
  tensor::Tensor output =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {dim}, alloc_cu);

  for (int i = 0; i < size; ++i) {
    weight.at<float>(i) = static_cast<float>(i);
  }
  weight.to_cuda();
  kernel::get_emb_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output,
                                                        token, nullptr);

  output.to_cpu();
  for (int i = 0; i < dim; ++i) {
    ASSERT_EQ(output.at<float>(i), 1024 + i);
  }
}

TEST(test_emb_cu, emb1_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t token = 4;
  int32_t dim = 512;
  int32_t size = 2048;

  tensor::Tensor input =
      tensor::Tensor::allocate(base::DataType::kDataTypeInt32, {1}, alloc_cpu);
  input.at<int32_t>(0) = 1;

  tensor::Tensor weight = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {token, dim}, alloc_cpu);
  tensor::Tensor output =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {dim}, alloc_cu);

  for (int i = 0; i < size; ++i) {
    weight.at<float>(i) = static_cast<float>(i);
  }
  weight.to_cuda();
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_emb_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output,
                                                        token, stream);

  output.to_cpu();
  for (int i = 0; i < dim; ++i) {
    ASSERT_EQ(output.at<float>(i), 512 + i);
  }

  cudaStreamDestroy(stream);
}
