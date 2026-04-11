#include "base/alloc.h"

#include <cuda_runtime_api.h>
#include <glog/logging.h>

namespace base {
void DeviceAllocator::memcpy(void *dst, const void *src, std::size_t size,
                             cudaMemcpyKind kind, cudaStream_t stream) const {
  CHECK_NE(src, nullptr) << "src is nullptr";
  CHECK_NE(dst, nullptr) << "dst is nullptr";
  CHECK_NE(size, 0) << "size is 0";

  if (kind == cudaMemcpyHostToHost) {
    std::memcpy(dst, src, size);
  } else if (kind == cudaMemcpyHostToDevice) {
    if (!stream) {
      cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    } else {
      cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
    }
  } else if (kind == cudaMemcpyDeviceToHost) {
    if (!stream) {
      cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    } else {
      cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
    }
  } else if (kind == cudaMemcpyDeviceToDevice) {
    if (!stream) {
      cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
    }
  } else {
    LOG(FATAL) << "Unknown memcpy kind: " << int(kind);
  }
}

void DeviceAllocator::memset_zero(void *ptr, size_t byte_size,
                                  cudaStream_t stream) {
  CHECK(device_type_ != base::DeviceType::kDeviceUnknown);
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    std::memset(ptr, 0, byte_size);
  } else {
    if (stream) {
      cudaMemsetAsync(ptr, 0, byte_size, stream);
    } else {
      cudaMemset(ptr, 0, byte_size);
    }
  }
}

}  // namespace base
