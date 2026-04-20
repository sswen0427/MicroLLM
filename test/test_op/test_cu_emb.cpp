#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "base/buffer.h"
#include "op/kernels/kernels_interface.h"

TEST(CudaEmbTest, NoStream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t token = 4;
  int32_t dim = 512;

  // init input
  tensor::Tensor input =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {1}, alloc_cpu);
  input.at<int32_t>(0) = 1;

  // init weight
  tensor::Tensor weight = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {token, dim}, alloc_cpu);
  for (int i = 0; i < token * dim; ++i) {
    weight.at<float>(i) = static_cast<float>(i);
  }
  weight.to_cuda();

  // init output
  tensor::Tensor output =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {dim}, alloc_cu);

  kernel::get_emb_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output,
                                                        token, nullptr);

  output.to_cpu();
  for (int i = 0; i < dim; ++i) {
    EXPECT_EQ(output.at<float>(i), 512 + i);
  }
}

TEST(CudaEmbTest, NoStream2) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t token = 4;
  int32_t dim = 512;
  int32_t size = 2048;

  // init input
  tensor::Tensor input =
      tensor::Tensor::allocate(base::DataType::kDataTypeInt32, {1}, alloc_cpu);
  input.at<int32_t>(0) = 2;

  // init weight
  tensor::Tensor weight = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {token, dim}, alloc_cpu);
  for (int i = 0; i < size; ++i) {
    weight.at<float>(i) = static_cast<float>(i);
  }
  weight.to_cuda();

  // init output
  tensor::Tensor output =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {dim}, alloc_cu);

  kernel::get_emb_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output,
                                                        token, nullptr);
  output.to_cpu();
  for (int i = 0; i < dim; ++i) {
    EXPECT_EQ(output.at<float>(i), 1024 + i);
  }
}

TEST(CudaEmbTest, Stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t token = 4;
  int32_t dim = 512;
  int32_t size = 2048;

  // init input
  tensor::Tensor input =
      tensor::Tensor::allocate(base::DataType::kDataTypeInt32, {1}, alloc_cpu);
  input.at<int32_t>(0) = 1;

  // init weight
  tensor::Tensor weight = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {token, dim}, alloc_cpu);
  for (int i = 0; i < size; ++i) {
    weight.at<float>(i) = static_cast<float>(i);
  }
  weight.to_cuda();

  // init output
  tensor::Tensor output =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {dim}, alloc_cu);

  // init stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_emb_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output,
                                                        token, stream);

  output.to_cpu();
  for (int i = 0; i < dim; ++i) {
    EXPECT_EQ(output.at<float>(i), 512 + i);
  }

  cudaStreamDestroy(stream);
}
