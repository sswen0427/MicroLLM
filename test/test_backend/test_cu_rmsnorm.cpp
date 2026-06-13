#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>

#include "cuda/rmsnorm_kernel.cuh"

namespace {

void RmsNormGolden(const tensor::Tensor &input, const tensor::Tensor &weight,
                   tensor::Tensor &output) {
  float square_sum = 0.0f;
  for (size_t i = 0; i < input.size(); ++i) {
    const float value = input.at<float>(i);
    square_sum += value * value;
  }
  const float mean_square = square_sum / static_cast<float>(input.size());
  const float scale = 1.0f / std::sqrt(mean_square + 1e-5f);
  for (size_t i = 0; i < input.size(); ++i) {
    output.at<float>(i) = input.at<float>(i) * scale * weight.at<float>(i);
  }
}

}  // namespace

TEST(RMSNormTest, NoStream) {
  int32_t size = 32 * 15;

  tensor::Tensor in_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCPU);
  tensor::Tensor wei_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCPU);
  tensor::Tensor out_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCPU);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.at<float>(i) = dist(mt);
    wei_cpu.at<float>(i) = dist(mt);
  }
  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);

  kernel::rmsnorm_kernel_cu(in_cu, wei_cu, out_cu, nullptr);
  out_cu.to_cpu();
  RmsNormGolden(in_cpu, wei_cpu, out_cpu);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.at<float>(i), out_cpu.at<float>(i), 1e-5f);
  }
}

TEST(RMSNormTest, Stream2) {
  int32_t size = 32;

  tensor::Tensor in_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCPU);
  tensor::Tensor wei_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCPU);
  tensor::Tensor out_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCPU);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.at<float>(i) = dist(mt);
    wei_cpu.at<float>(i) = dist(mt);
  }

  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::rmsnorm_kernel_cu(in_cu, wei_cu, out_cu, stream);
  out_cu.to_cpu();

  RmsNormGolden(in_cpu, wei_cpu, out_cpu);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.at<float>(i), out_cpu.at<float>(i), 1e-5f);
  }
  cudaStreamDestroy(stream);
}

TEST(RMSNormTest, Stream3) {
  int32_t size = 32 * 151 * 15;

  tensor::Tensor in_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCPU);
  tensor::Tensor wei_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCPU);
  tensor::Tensor out_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {size}, base::DeviceType::kDeviceCPU);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.at<float>(i) = dist(mt);
    wei_cpu.at<float>(i) = dist(mt);
  }

  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::rmsnorm_kernel_cu(in_cu, wei_cu, out_cu, stream);
  out_cu.to_cpu();

  RmsNormGolden(in_cpu, wei_cpu, out_cpu);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.at<float>(i), out_cpu.at<float>(i), 1e-5f);
  }
  cudaStreamDestroy(stream);
}
