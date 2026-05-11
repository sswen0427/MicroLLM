#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "base/buffer.h"
#include "op/kernels/kernels_interface.h"
TEST(ScaleTest, Nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  int32_t size = 32 * 151;

  tensor::Tensor t1 =
      tensor::Tensor::allocate(base::DataType::kDataTypeFp32, {size}, alloc_cu);
  std::vector<float> vec(size, 2.f);
  cudaMemcpy(t1.get_buffer()->ptr(), vec.data(), size * sizeof(float),
             cudaMemcpyHostToHost);

  kernel::get_scale_kernel(base::DeviceType::kDeviceCPU)(0.5f, t1, nullptr);
  cudaDeviceSynchronize();

  t1.to_cpu();
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(t1.at<float>(i), 1.f);
  }
}
