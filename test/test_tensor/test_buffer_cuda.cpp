#include <gtest/gtest.h>

#include "base/buffer.h"

TEST(BufferCudaTest, CopyBetweenCPUAndCUDABuffers) {
  constexpr int buffer_size = 32;

  auto test_cpu_buffer = [&](const base::Buffer& buffer) {
    for (int i = 0; i < buffer_size; ++i) {
      EXPECT_EQ(static_cast<int*>(buffer.ptr())[i], i);
    }
  };

  auto test_cuda_buffer = [&](const base::Buffer& buffer) {
    base::Buffer buffer_cpu(buffer_size * sizeof(int),
                            base::DeviceType::kDeviceCPU);
    buffer_cpu.copy_from(buffer);
    test_cpu_buffer(buffer_cpu);
  };

  base::Buffer buffer_cpu1(buffer_size * sizeof(int),
                           base::DeviceType::kDeviceCPU);
  int* cpu_ptr1 = static_cast<int*>(buffer_cpu1.ptr());
  for (int i = 0; i < buffer_size; ++i) {
    cpu_ptr1[i] = i;
  }

  base::Buffer buffer_cuda1(buffer_size * sizeof(int),
                            base::DeviceType::kDeviceCUDA);
  buffer_cuda1.copy_from(buffer_cpu1);
  test_cuda_buffer(buffer_cuda1);

  base::Buffer buffer_cuda2(buffer_size * sizeof(int),
                            base::DeviceType::kDeviceCUDA);
  buffer_cuda2.copy_from(buffer_cuda1);
  test_cuda_buffer(buffer_cuda2);

  base::Buffer buffer_cpu2(buffer_size * sizeof(int),
                           base::DeviceType::kDeviceCPU);
  buffer_cpu2.copy_from(buffer_cuda2);
  test_cpu_buffer(buffer_cpu2);
}
