#ifndef MICROLLM_INCLUDE_BASE_ALLOC_H
#define MICROLLM_INCLUDE_BASE_ALLOC_H

#include "base.h"

namespace base {
class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceType device_type)
      : device_type_(device_type) {}

  virtual void release(void *ptr) = 0;

  virtual void *allocate(std::size_t size) = 0;

 private:
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  void release(void *ptr) override;

  void *allocate(std::size_t size) override;
};
}  // namespace base

#endif
