#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <random>

#include "base/buffer.h"
#include "op/kernels/kernels_interface.h"

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

  kernel::get_swiglu_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu,
                                                           out_cu, nullptr);
  out_cu.to_cpu();

  kernel::get_swiglu_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu,
                                                          out_cpu, nullptr);

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

  kernel::get_swiglu_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu,
                                                           out_cu, stream);
  out_cu.to_cpu();

  kernel::get_swiglu_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu,
                                                          out_cpu, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.at<float>(i), out_cpu.at<float>(i), 1e-5f);
  }
  cudaStreamDestroy(stream);
}