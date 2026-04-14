#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tensor/tensor.h>

#include "base/base.h"
#include "base/buffer.h"

TEST(TensorTest, TOCPU) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cu = tensor::Tensor::allocate(
      base::DataType::kDataTypeInt32, {32, 32}, alloc_cu);
  CHECK(!t1_cu.is_empty());

  int array[32 * 32];
  for (int i = 0; i < 32 * 32; i++) {
    array[i] = i;
  }
  base::Buffer buffer_cpu(32 * 32 * 4, nullptr, array);
  t1_cu.get_buffer().get()->copy_from(buffer_cpu);
  t1_cu.to_cpu();

  CHECK(t1_cu.device_type() == base::DeviceType::kDeviceCPU);

  int* cpu_ptr = t1_cu.ptr<int>();
  for (int i = 0; i < 32 * 32; ++i) {
    CHECK_EQ(*(cpu_ptr + i), i);
  }
}

TEST(TensorTest, TOCUDA) {
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cpu = tensor::Tensor::allocate(
      base::DataType::kDataTypeInt32, {32, 32}, alloc_cpu);
  CHECK(!t1_cpu.is_empty());

  int array[32 * 32];
  for (int i = 0; i < 32 * 32; i++) {
    array[i] = i;
  }
  base::Buffer buffer_cpu(32 * 32 * 4, nullptr, array);
  t1_cpu.get_buffer().get()->copy_from(buffer_cpu);
  t1_cpu.to_cuda();

  int expected_array[32 * 32];
  cudaMemcpy(expected_array, t1_cpu.ptr<int>(), 32 * 32 * 4,
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    CHECK_EQ(expected_array[i], i);
  }
}

// TEST(test_tensor, clone_cuda) {
//   using namespace base;
//   auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
//   tensor::Tensor t1_cu(DataType::kDataTypeFp32, 32, 32, true, alloc_cu);
//   ASSERT_EQ(t1_cu.is_empty(), false);
//   set_value_cu(t1_cu.ptr<float>(), 32 * 32, 1.f);
//
//   tensor::Tensor t2_cu = t1_cu.clone();
//   float* p2 = new float[32 * 32];
//   cudaMemcpy(p2, t2_cu.ptr<float>(), sizeof(float) * 32 * 32,
//   cudaMemcpyDeviceToHost); for (int i = 0; i < 32 * 32; ++i) {
//     ASSERT_EQ(p2[i], 1.f);
//   }
//
//   cudaMemcpy(p2, t1_cu.ptr<float>(), sizeof(float) * 32 * 32,
//   cudaMemcpyDeviceToHost); for (int i = 0; i < 32 * 32; ++i) {
//     ASSERT_EQ(p2[i], 1.f);
//   }
//
//   ASSERT_EQ(t2_cu.data_type(), base::DataType::kDataTypeFp32);
//   ASSERT_EQ(t2_cu.size(), 32 * 32);
//
//   t2_cu.to_cpu();
//   std::memcpy(p2, t2_cu.ptr<float>(), sizeof(float) * 32 * 32);
//   for (int i = 0; i < 32 * 32; ++i) {
//     ASSERT_EQ(p2[i], 1.f);
//   }
//   delete[] p2;
// }
//
// TEST(test_tensor, clone_cpu) {
//   using namespace base;
//   auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
//   tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
//   ASSERT_EQ(t1_cpu.is_empty(), false);
//   for (int i = 0; i < 32 * 32; ++i) {
//     t1_cpu.index<float>(i) = 1.f;
//   }
//
//   tensor::Tensor t2_cpu = t1_cpu.clone();
//   float* p2 = new float[32 * 32];
//   std::memcpy(p2, t2_cpu.ptr<float>(), sizeof(float) * 32 * 32);
//   for (int i = 0; i < 32 * 32; ++i) {
//     ASSERT_EQ(p2[i], 1.f);
//   }
//
//   std::memcpy(p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32);
//   for (int i = 0; i < 32 * 32; ++i) {
//     ASSERT_EQ(p2[i], 1.f);
//   }
//   delete[] p2;
// }
//
