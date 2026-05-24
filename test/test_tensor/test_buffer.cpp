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

TEST(BufferTest, CopyBetweenCPUBuffers) {
  constexpr int buffer_size = 32;
  base::Buffer buffer_cpu1(buffer_size * sizeof(int),
                           base::DeviceType::kDeviceCPU);
  int* cpu_ptr1 = static_cast<int*>(buffer_cpu1.ptr());
  for (int i = 0; i < buffer_size; ++i) {
    cpu_ptr1[i] = i;
  }

  base::Buffer buffer_cpu2(buffer_size * sizeof(int),
                           base::DeviceType::kDeviceCPU);
  buffer_cpu2.copy_from(buffer_cpu1);

  for (int i = 0; i < buffer_size; ++i) {
    EXPECT_EQ(static_cast<int*>(buffer_cpu2.ptr())[i], i);
  }
}
