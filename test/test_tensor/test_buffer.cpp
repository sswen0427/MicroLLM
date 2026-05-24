#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "base/buffer.h"

TEST(BufferTest, Allocate) {
  base::Buffer buffer(32, base::DeviceType::kDeviceCPU);
  EXPECT_NE(buffer.ptr(), nullptr);
}

TEST(BufferTest, UseExternal) {
  float array[32];
  base::Buffer buffer(sizeof(array), array, base::DeviceType::kDeviceCPU);
  EXPECT_TRUE(buffer.is_external());
}

TEST(BufferTest, CopyToExternalCPUBuffer) {
  constexpr int buffer_size = 4;
  base::Buffer src(buffer_size * sizeof(int), base::DeviceType::kDeviceCPU);
  int* src_data = static_cast<int*>(src.ptr());
  for (int i = 0; i < buffer_size; ++i) {
    src_data[i] = i;
  }

  int dst_data[buffer_size] = {};
  base::Buffer dst(sizeof(dst_data), dst_data, base::DeviceType::kDeviceCPU);

  dst.copy_from(src);

  for (int i = 0; i < buffer_size; ++i) {
    EXPECT_EQ(dst_data[i], i);
  }
}

TEST(BufferTest, Memcpy) {
  int buffer_size = 32;

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

  // Step1: Create a CPU buffer with size 32
  base::Buffer buffer_cpu1(buffer_size * sizeof(int),
                           base::DeviceType::kDeviceCPU);
  int* cpu_ptr1 = static_cast<int*>(buffer_cpu1.ptr());
  for (int i = 0; i < buffer_size; ++i) {
    cpu_ptr1[i] = i;
  }

  // Step2: Copy buffer_cpu1 to buffer_cpu2
  base::Buffer buffer_cpu2(buffer_size * sizeof(int),
                           base::DeviceType::kDeviceCPU);
  buffer_cpu2.copy_from(buffer_cpu1);
  test_cpu_buffer(buffer_cpu2);

  // Step3: Copy buffer_cpu2 to buffer_cuda1
  base::Buffer buffer_cuda1(buffer_size * sizeof(int),
                            base::DeviceType::kDeviceCUDA);
  buffer_cuda1.copy_from(buffer_cpu2);
  test_cuda_buffer(buffer_cuda1);

  // Step4: Copy buffer_cuda1 to buffer_cuda2
  base::Buffer buffer_cuda2(buffer_size * sizeof(int),
                            base::DeviceType::kDeviceCUDA);
  buffer_cuda2.copy_from(buffer_cuda1);
  test_cuda_buffer(buffer_cuda2);

  // Step5: Copy buffer_cuda2 to buffer_cpu3
  base::Buffer buffer_cpu3(buffer_size * sizeof(int),
                           base::DeviceType::kDeviceCPU);
  buffer_cpu3.copy_from(buffer_cuda2);
  test_cpu_buffer(buffer_cpu3);
}
