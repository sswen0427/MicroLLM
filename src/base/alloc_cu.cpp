#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include "base/alloc.h"
#include "base/cuda_check.h"

namespace base {

CUDADeviceAllocator::CUDADeviceAllocator()
    : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void* CUDADeviceAllocator::allocate(std::size_t byte_size) const {
  CHECK(byte_size > 0) << "CUDADeviceAllocator::allocate(): byte_size is 0";

  void* ptr = nullptr;
  CHECK_CUDA(cudaMalloc(&ptr, byte_size));
  return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
  CHECK(ptr) << "CUDADeviceAllocator::release(): ptr is nullptr";
  CHECK_CUDA(cudaFree(ptr));
}
}  // namespace base
