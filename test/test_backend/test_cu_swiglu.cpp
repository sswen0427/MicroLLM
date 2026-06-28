#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>

#include "cuda/swiglu.cuh"

namespace {

void SwiGluGolden(const tensor::Tensor &gate, const tensor::Tensor &up,
                  tensor::Tensor &output) {
  for (size_t i = 0; i < gate.size(); ++i) {
    const float value = gate.at<float>(i);
    output.at<float>(i) = value / (1.0f + std::exp(-value)) * up.at<float>(i);
  }
}

}  // namespace

TEST(SwiGLUTest, NoStream) {
  int32_t size = 32 * 151;

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

  kernel::SwiGluCuda(in_cu, wei_cu, out_cu, nullptr);
  out_cu.to_cpu();

  SwiGluGolden(in_cpu, wei_cpu, out_cpu);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.at<float>(i), out_cpu.at<float>(i), 1e-5f);
  }
}

TEST(SwiGLUTest, Stream) {
  int32_t size = 32 * 151;

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

  kernel::SwiGluCuda(in_cu, wei_cu, out_cu, stream);
  out_cu.to_cpu();

  SwiGluGolden(in_cpu, wei_cpu, out_cpu);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.at<float>(i), out_cpu.at<float>(i), 1e-5f);
  }
  cudaStreamDestroy(stream);
}
