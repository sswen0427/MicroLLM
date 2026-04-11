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

TEST(BufferTest, CUDAMemcpy1) {
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int array_cpu[32];
  for (int i = 0; i < 32; ++i) {
    array_cpu[i] = i;
  }
  base::Buffer buffer_cpu(32 * sizeof(int), nullptr, array_cpu);
  buffer_cpu.set_device_type(base::DeviceType::kDeviceCPU);
  EXPECT_EQ(buffer_cpu.is_external(), true);

  base::Buffer buffer_cuda(32 * sizeof(int), alloc_cu);
  buffer_cuda.copy_from(buffer_cpu);

  float array_cuda[32];
  cudaMemcpy(array_cuda, buffer_cuda.ptr(), 32 * sizeof(float),cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(array_cuda[i], i);
  }
}
//
// TEST(test_buffer, cuda_memcpy2) {
//   using namespace base;
//   auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
//   auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
//
//   int32_t size = 32;
//   float* ptr = new float[size];
//   for (int i = 0; i < size; ++i) {
//     ptr[i] = float(i);
//   }
//   Buffer buffer(size * sizeof(float), nullptr, ptr, true);
//   buffer.set_device_type(DeviceType::kDeviceCPU);
//   ASSERT_EQ(buffer.is_external(), true);
//
//   // cpu to cuda
//   Buffer cu_buffer(size * sizeof(float), alloc_cu);
//   cu_buffer.copy_from(buffer);
//
//   float* ptr2 = new float[size];
//   cudaMemcpy(ptr2, cu_buffer.ptr(), sizeof(float) * size,
//   cudaMemcpyDeviceToHost); for (int i = 0; i < size; ++i) {
//     ASSERT_EQ(ptr2[i], float(i));
//   }
//
//   delete[] ptr;
//   delete[] ptr2;
// }
//
// TEST(test_buffer, cuda_memcpy3) {
//   using namespace base;
//   auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
//   auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
//
//   int32_t size = 32;
//   Buffer cu_buffer1(size * sizeof(float), alloc_cu);
//   Buffer cu_buffer2(size * sizeof(float), alloc_cu);
//
//   set_value_cu((float*)cu_buffer2.ptr(), size);
//   // cu to cu
//   ASSERT_EQ(cu_buffer1.device_type(), DeviceType::kDeviceCUDA);
//   ASSERT_EQ(cu_buffer2.device_type(), DeviceType::kDeviceCUDA);
//
//   cu_buffer1.copy_from(cu_buffer2);
//
//   float* ptr2 = new float[size];
//   cudaMemcpy(ptr2, cu_buffer1.ptr(), sizeof(float) * size,
//   cudaMemcpyDeviceToHost); for (int i = 0; i < size; ++i) {
//     ASSERT_EQ(ptr2[i], 1.f);
//   }
//   delete[] ptr2;
// }
//
// TEST(test_buffer, cuda_memcpy4) {
//   using namespace base;
//   auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
//   auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
//
//   int32_t size = 32;
//   Buffer cu_buffer1(size * sizeof(float), alloc_cu);
//   Buffer cu_buffer2(size * sizeof(float), alloc);
//   ASSERT_EQ(cu_buffer1.device_type(), DeviceType::kDeviceCUDA);
//   ASSERT_EQ(cu_buffer2.device_type(), DeviceType::kDeviceCPU);
//
//   // cu to cpu
//   set_value_cu((float*)cu_buffer1.ptr(), size);
//   cu_buffer2.copy_from(cu_buffer1);
//
//   float* ptr2 = (float*)cu_buffer2.ptr();
//   for (int i = 0; i < size; ++i) {
//     ASSERT_EQ(ptr2[i], 1.f);
//   }
// }