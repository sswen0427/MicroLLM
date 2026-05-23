#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "base/buffer.h"
#include "op/kernels/kernels_interface.h"
TEST(CudaAddTest, NoStream) {
  int32_t size = 32 * 151;
  tensor::Tensor t1 =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCUDA);
  tensor::Tensor t2 =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCUDA);
  tensor::Tensor out =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCUDA);

  std::vector<float> vec_2(size, 2.f);
  std::vector<float> vec_3(size, 3.f);
  cudaMemcpy(t1.data<float>(), vec_2.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(t2.data<float>(), vec_3.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);

  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, nullptr);
  cudaDeviceSynchronize();
  std::vector<float> output(size);
  cudaMemcpy(output.data(), out.data<float>(), size * sizeof(float),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(output[i], 5.f);
  }
}

TEST(CudaAddTest, Stream) {
  int32_t size = 32 * 151;
  tensor::Tensor t1 =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCUDA);
  tensor::Tensor t2 =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCUDA);
  tensor::Tensor out =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCUDA);
  std::vector<float> vec_2(size, 2.f);
  std::vector<float> vec_3(size, 3.f);
  cudaMemcpy(t1.data<float>(), vec_2.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(t2.data<float>(), vec_3.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, stream);
  cudaDeviceSynchronize();
  std::vector<float> output(size);
  cudaMemcpy(output.data(), out.data<float>(), size * sizeof(float),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(output[i], 5.f);
  }
  cudaStreamDestroy(stream);
}

TEST(CudaAddTest, Align) {
  int32_t size = 32 * 151 * 13;
  tensor::Tensor t1 =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCUDA);
  tensor::Tensor t2 =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCUDA);
  tensor::Tensor out =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCUDA);

  std::vector<float> vec_2(size, 2.1f);
  std::vector<float> vec_3(size, 3.3f);
  cudaMemcpy(t1.data<float>(), vec_2.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(t2.data<float>(), vec_3.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);

  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, nullptr);
  cudaDeviceSynchronize();
  std::vector<float> output(size);
  cudaMemcpy(output.data(), out.data<float>(), size * sizeof(float),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    EXPECT_NEAR(output[i], 5.4f, 0.1f);
  }
}
