#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "base/buffer.h"
#include "op/kernels/kernels_interface.h"
TEST(test_scale_cu, scale1_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  int32_t size = 32 * 151;

  tensor::Tensor t1 =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, alloc_cu);
  std::vector<float> vec_2(size, 2.f);
  cudaMemcpy(t1.get_buffer()->ptr(), vec_2.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);

  kernel::get_scale_kernel(base::DeviceType::kDeviceCUDA)(0.5f, t1, nullptr);
  cudaDeviceSynchronize();

  t1.to_cpu();
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(t1.at<float>(i), 1.f);
  }
}

TEST(test_scale_cu, scale1_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  int32_t size = 32 * 151;

  tensor::Tensor t1 =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, alloc_cu);
  std::vector<float> vec_2(size, 2.f);
  cudaMemcpy(t1.get_buffer()->ptr(), vec_2.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_scale_kernel(base::DeviceType::kDeviceCUDA)(0.4f, t1, nullptr);
  cudaDeviceSynchronize();

  t1.to_cpu();
  cudaStreamDestroy(stream);

  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(t1.at<float>(i), 0.8f);
  }
}
