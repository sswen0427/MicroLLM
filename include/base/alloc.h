#ifndef MICROLLM_INCLUDE_BASE_ALLOC_H
#define MICROLLM_INCLUDE_BASE_ALLOC_H

#include <driver_types.h>

#include <map>
#include <memory>
#include <vector>

#include "base.h"

namespace base {

class DeviceAllocator {
 public:
  explicit DeviceAllocator(const DeviceType device_type)
      : device_type_(device_type) {}

  [[nodiscard]] virtual DeviceType device_type() const { return device_type_; }

  virtual void *allocate(std::size_t size) const = 0;

  virtual void release(void *ptr) const = 0;

  virtual void memcpy(void *dst, const void *src, std::size_t size,
                      cudaMemcpyKind kind, cudaStream_t stream) const;

  virtual void memset_zero(void *ptr, size_t byte_size, cudaStream_t stream);

 private:
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  void *allocate(size_t byte_size) const override;

  void release(void *ptr) const override;
};

struct CUDAMemoryBuffer {
  void *data;
  std::size_t size;
  bool busy;
};

class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator();

  void *allocate(size_t byte_size) const override;

  void release(void *ptr) const override;

 private:
  mutable std::map<int, size_t> no_busy_cnt_;
  mutable std::map<int, std::vector<CUDAMemoryBuffer>> big_buffers_map_;
  mutable std::map<int, std::vector<CUDAMemoryBuffer>> cuda_buffers_map_;
};

class CPUDeviceAllocatorFactory {
 public:
  static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CPUDeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CPUDeviceAllocator> instance;
};

class CUDADeviceAllocatorFactory {
 public:
  static std::shared_ptr<CUDADeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CUDADeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CUDADeviceAllocator> instance;
};

}  // namespace base

#endif
