#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

#include "base/types.h"

namespace base {

class DeviceAllocator {
 public:
  explicit DeviceAllocator(const DeviceType device_type)
      : device_type_(device_type) {}

  virtual ~DeviceAllocator() = default;

  [[nodiscard]] DeviceType device_type() const { return device_type_; }

  [[nodiscard]] virtual void *allocate(std::size_t size) const = 0;

  virtual void release(void *ptr) const = 0;

 private:
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  [[nodiscard]] void *allocate(std::size_t byte_size) const override;

  void release(void *ptr) const override;
};

class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator();

  [[nodiscard]] void *allocate(std::size_t byte_size) const override;

  void release(void *ptr) const override;
};

std::shared_ptr<DeviceAllocator> GetDeviceAllocator(DeviceType device_type);

void CopyMemory(void *dst, const void *src, std::size_t size,
                cudaMemcpyKind kind, cudaStream_t stream = nullptr);

void MemsetZero(DeviceType device_type, void *ptr, std::size_t byte_size,
                cudaStream_t stream = nullptr);

}  // namespace base
