#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "base/alloc.h"
#include "base/buffer.h"

TEST(BufferTest, Allocate) {
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  base::Buffer buffer(32, alloc);
  EXPECT_NE(buffer.ptr(), nullptr);
}

TEST(BufferTest, UseExternal) {
  float array[32];
  base::Buffer buffer(32, nullptr, array);
  EXPECT_TRUE(buffer.is_external());
}

TEST(BufferTest, Memcpy) {
  int buffer_size = 32;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  auto test_cpu_buffer = [&](const base::Buffer& buffer){
    for (int i = 0; i < buffer_size; ++i) {
      EXPECT_EQ(static_cast<int*>(buffer.ptr())[i], i);
    }
  };

  auto test_cuda_buffer = [&](const base::Buffer& buffer){
    base::Buffer buffer_cpu(32, alloc);
    buffer_cpu.copy_from(buffer);
    test_cpu_buffer(buffer_cpu);
  };

  // Step1: Create a CPU buffer with size 32
  base::Buffer buffer_cpu1(buffer_size * sizeof(int), alloc);
  int* cpu_ptr1 = static_cast<int*>(buffer_cpu1.ptr());
  for (int i = 0; i < buffer_size; ++i) {
    cpu_ptr1[i] = i;
  }

  // Step2: Copy buffer_cpu1 to buffer_cpu2
  base::Buffer buffer_cpu2(buffer_size * sizeof(int), alloc);
  buffer_cpu2.copy_from(buffer_cpu1);
  test_cpu_buffer(buffer_cpu2);

  // Step3: Copy buffer_cpu2 to buffer_cuda1
  base::Buffer buffer_cuda1(buffer_size * sizeof(int), alloc_cu);
  buffer_cuda1.copy_from(buffer_cpu2);
  test_cuda_buffer(buffer_cuda1);

  // Step4: Copy buffer_cuda1 to buffer_cuda2
  base::Buffer buffer_cuda2(buffer_size * sizeof(int), alloc_cu);
  buffer_cuda2.copy_from(buffer_cuda1);
  test_cuda_buffer(buffer_cuda2);

  // Step5: Copy buffer_cuda2 to buffer_cpu3
  base::Buffer buffer_cpu3(buffer_size * sizeof(int), alloc);
  buffer_cpu3.copy_from(buffer_cuda2);
  test_cpu_buffer(buffer_cpu3);
}