#include "base/alloc.h"

#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include <cstring>

#include "base/cuda_check.h"

namespace base {

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

void DeviceAllocator::memcpy(void *dst, const void *src, std::size_t size,
                             cudaMemcpyKind kind, cudaStream_t stream) const {
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

void DeviceAllocator::memset_zero(void *ptr, std::size_t byte_size,
                                  cudaStream_t stream) const {
  CHECK_NE(ptr, nullptr) << "ptr is nullptr";
  CHECK_NE(byte_size, 0) << "byte_size is 0";
  CHECK(device_type_ != base::DeviceType::kDeviceUnknown);
  if (device_type_ == base::DeviceType::kDeviceCPU) {
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
