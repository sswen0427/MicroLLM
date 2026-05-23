#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include "base/alloc.h"

namespace base {
namespace {

void CheckCuda(cudaError_t state, const char* operation) {
  CHECK_EQ(state, cudaSuccess)
      << operation << " failed: " << cudaGetErrorString(state);
}

}  // namespace

CUDADeviceAllocator::CUDADeviceAllocator()
    : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void* CUDADeviceAllocator::allocate(std::size_t byte_size) const {
  CHECK(byte_size > 0) << "CUDADeviceAllocator::allocate(): byte_size is 0";

  void* ptr = nullptr;
  const cudaError_t state = cudaMalloc(&ptr, byte_size);
  if (state != cudaSuccess) {
    LOG(ERROR) << "cudaMalloc failed! error code: "
               << cudaGetErrorString(state);
    return nullptr;
  }
  return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
  CHECK(ptr) << "CUDADeviceAllocator::release(): ptr is nullptr";
  CheckCuda(cudaFree(ptr), "cudaFree");
}
}  // namespace base
