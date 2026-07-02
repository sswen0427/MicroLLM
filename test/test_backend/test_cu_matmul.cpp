#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "cuda/matmul.cuh"

TEST(CudaMatmulTest, RunCUDA) {
  tensor::Tensor input = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {4}, base::DeviceType::kDeviceCPU);
  tensor::Tensor weight = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {4, 4}, base::DeviceType::kDeviceCPU);

  input.at<float>(0) = float(1);
  input.at<float>(1) = float(1);
  input.at<float>(2) = float(-1);
  input.at<float>(3) = float(0);

  for (int i = 1; i <= 16; ++i) {
    weight.at<float>(i - 1) = float(i);
  }

  input.to_cuda();
  weight.to_cuda();

  tensor::Tensor out_cu = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {4}, base::DeviceType::kDeviceCUDA);

  kernel::MatmulCuda(input, weight, out_cu, 1.f, nullptr);

  tensor::Tensor out_cpu = out_cu.clone();
  out_cpu.to_cpu();

  EXPECT_EQ(out_cpu.at<float>(0), 0);
  EXPECT_EQ(out_cpu.at<float>(1), 4);
  EXPECT_EQ(out_cpu.at<float>(2), 8);
  EXPECT_EQ(out_cpu.at<float>(3), 12);
}

TEST(CudaMatmulTest, Stream) {
  tensor::Tensor input = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {4}, base::DeviceType::kDeviceCPU);
  tensor::Tensor weight = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {4, 4}, base::DeviceType::kDeviceCPU);

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

  tensor::Tensor out_cu = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {4}, base::DeviceType::kDeviceCUDA);
  tensor::Tensor out_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {4}, base::DeviceType::kDeviceCPU);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::MatmulCuda(input, weight, out_cu, 1.f, stream);

  for (int row = 0; row < 4; ++row) {
    float sum = 0.0f;
    for (int col = 0; col < 4; ++col) {
      sum += weight_cpu.at<float>(row * 4 + col) * input_cpu.at<float>(col);
    }
    out_cpu.at<float>(row) = sum;
  }

  out_cu.to_cpu();
  for (int i = 0; i < out_cu.size(); ++i) {
    EXPECT_EQ(out_cu.at<float>(i), out_cpu.at<float>(i));
  }
  cudaStreamDestroy(stream);
}

TEST(CudaMatmulTest, SupportsNonMultipleOfFourColumns) {
  constexpr int kRows = 3;
  constexpr int kCols = 5;
  tensor::Tensor input = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {kCols}, base::DeviceType::kDeviceCPU);
  tensor::Tensor weight =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {kRows, kCols},
                               base::DeviceType::kDeviceCPU);

  for (int i = 0; i < kCols; ++i) {
    input.at<float>(i) = static_cast<float>(i + 1);
  }
  for (int i = 0; i < kRows * kCols; ++i) {
    weight.at<float>(i) = static_cast<float>(i + 1);
  }

  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();
  input.to_cuda();
  weight.to_cuda();

  tensor::Tensor out_cu = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {kRows}, base::DeviceType::kDeviceCUDA);

  kernel::MatmulCuda(input, weight, out_cu, 1.f, nullptr);

  out_cu.to_cpu();
  for (int row = 0; row < kRows; ++row) {
    float expected = 0.0f;
    for (int col = 0; col < kCols; ++col) {
      expected +=
          weight_cpu.at<float>(row * kCols + col) * input_cpu.at<float>(col);
    }
    EXPECT_EQ(out_cu.at<float>(row), expected);
  }
}
