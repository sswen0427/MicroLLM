#include "base/alloc.h"

#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include <cstring>
#include <cstdlib>

#include "base/cuda_check.h"

namespace base {

CPUDeviceAllocator::CPUDeviceAllocator()
    : DeviceAllocator(DeviceType::kDeviceCPU) {}

void* CPUDeviceAllocator::allocate(std::size_t byte_size) const {
  CHECK(byte_size > 0) << "CPUDeviceAllocator::allocate(): byte_size is 0";
  const size_t alignment = (byte_size >= 1024) ? 32 : 16;
  const size_t aligned_size = (byte_size + alignment - 1) & ~(alignment - 1);
  void* data = std::aligned_alloc(alignment, aligned_size);
  if (data == nullptr) {
    LOG(ERROR) << "std::aligned_alloc failed! "
               << "(alignment: " << alignment
               << ", original size: " << byte_size
               << ", padded size: " << aligned_size << ")";
    return nullptr;
  }
  return data;
}

void CPUDeviceAllocator::release(void* ptr) const {
  if (ptr) {
    std::free(ptr);
  }
}

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

std::shared_ptr<DeviceAllocator> GetDeviceAllocator(DeviceType device_type) {
  static std::shared_ptr<DeviceAllocator> cpu_allocator =
      std::make_shared<CPUDeviceAllocator>();
  static std::shared_ptr<DeviceAllocator> cuda_allocator =
      std::make_shared<CUDADeviceAllocator>();

  if (device_type == DeviceType::kDeviceCPU) {
    return cpu_allocator;
  }
  if (device_type == DeviceType::kDeviceCUDA) {
    return cuda_allocator;
  }
  LOG(FATAL) << "Unknown device type.";
  return nullptr;
}

void CopyMemory(void *dst, const void *src, std::size_t size,
                cudaMemcpyKind kind, cudaStream_t stream) {
  CHECK_NE(src, nullptr) << "src is nullptr";
  CHECK_NE(dst, nullptr) << "dst is nullptr";
  CHECK_NE(size, 0) << "size is 0";

  if (kind == cudaMemcpyHostToHost) {
    std::memcpy(dst, src, size);
  } else if (kind == cudaMemcpyHostToDevice) {
    if (!stream) {
      CHECK_CUDA(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    } else {
      CHECK_CUDA(
          cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
    }
  } else if (kind == cudaMemcpyDeviceToHost) {
    if (!stream) {
      CHECK_CUDA(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    } else {
      CHECK_CUDA(
          cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
    }
  } else if (kind == cudaMemcpyDeviceToDevice) {
    if (!stream) {
      CHECK_CUDA(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    } else {
      CHECK_CUDA(
          cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
    }
  } else {
    LOG(FATAL) << "Unknown memcpy kind: " << int(kind);
  }
}

void MemsetZero(DeviceType device_type, void *ptr, std::size_t byte_size,
                cudaStream_t stream) {
  CHECK_NE(ptr, nullptr) << "ptr is nullptr";
  CHECK_NE(byte_size, 0) << "byte_size is 0";
  CHECK(device_type != base::DeviceType::kDeviceUnknown);
  if (device_type == base::DeviceType::kDeviceCPU) {
    std::memset(ptr, 0, byte_size);
  } else {
    if (stream) {
      CHECK_CUDA(cudaMemsetAsync(ptr, 0, byte_size, stream));
    } else {
      CHECK_CUDA(cudaMemset(ptr, 0, byte_size));
    }
  }
}

}  // namespace base
