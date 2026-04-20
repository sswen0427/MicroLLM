#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "base/buffer.h"
#include "op/kernels/kernels_interface.h"
TEST(test_add_cu, add1_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor t1 =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, alloc_cu);
  tensor::Tensor t2 =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, alloc_cu);
  tensor::Tensor out =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, alloc_cu);

  std::vector<float> vec_2(size, 2.f);
  std::vector<float> vec_3(size, 3.f);
  cudaMemcpy(t1.get_buffer()->ptr(), vec_2.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(t2.get_buffer()->ptr(), vec_3.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);

  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, nullptr);
  cudaDeviceSynchronize();
  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output[i], 5.f);
  }

  delete[] output;
}

TEST(test_add_cu, add1_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor t1 =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, alloc_cu);
  tensor::Tensor t2 =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, alloc_cu);
  tensor::Tensor out =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, alloc_cu);
  std::vector<float> vec_2(size, 2.f);
  std::vector<float> vec_3(size, 3.f);
  cudaMemcpy(t1.get_buffer()->ptr(), vec_2.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(t2.get_buffer()->ptr(), vec_3.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, stream);
  cudaDeviceSynchronize();
  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output[i], 5.f);
  }
  cudaStreamDestroy(stream);
  delete[] output;
}

TEST(test_add_cu, add_align1) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151 * 13;

  tensor::Tensor t1 =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, alloc_cu);
  tensor::Tensor t2 =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, alloc_cu);
  tensor::Tensor out =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, alloc_cu);

  std::vector<float> vec_2(size, 2.1f);
  std::vector<float> vec_3(size, 3.3f);
  cudaMemcpy(t1.get_buffer()->ptr(), vec_2.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(t2.get_buffer()->ptr(), vec_3.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);

  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, nullptr);
  cudaDeviceSynchronize();
  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(output[i], 5.4f, 0.1f);
  }

  delete[] output;
}