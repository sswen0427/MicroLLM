#ifndef MICROLLM_INCLUDE_BASE_ALLOC_H
#define MICROLLM_INCLUDE_BASE_ALLOC_H

#include <map>
#include <memory>
#include <vector>

#include "base.h"

namespace base {

enum class MemcpyType {
  kMemcpyHostToHost,
  kMemcpyHostToDevice,
  kMemcpyDeviceToHost,
  kMemcpyDeviceToDevice,
};

class DeviceAllocator {
 public:
  explicit DeviceAllocator(const DeviceType device_type)
      : device_type_(device_type) {}

  [[nodiscard]] virtual DeviceType device_type() const { return device_type_; }

  virtual void *allocate(std::size_t size) = 0;

  virtual void release(void *ptr) = 0;

  virtual void memcpy(void *dst, const void *src, std::size_t size,
                      MemcpyType type = MemcpyType::kMemcpyHostToHost,
                      void *stream = nullptr, bool need_sync = false);

  virtual void memset_zero(void *ptr, std::size_t size);

 private:
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  void *allocate(std::size_t size) override;

  void release(void *ptr) override;
};

struct CUDAMemoryBuffer {
  void *data;
  std::size_t size;
  bool busy;
};

class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator();

  void *allocate(std::size_t size) override;

  void release(void *ptr) override;

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
