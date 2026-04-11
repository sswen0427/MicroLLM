#include <glog/logging.h>

#include "base/alloc.h"
#include "base/base.h"
namespace base {
CPUDeviceAllocator::CPUDeviceAllocator()
    : DeviceAllocator(DeviceType::kDeviceCPU) {}

void* CPUDeviceAllocator::allocate(size_t byte_size) const {
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
    free(ptr);
  }
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance =
    nullptr;
}  // namespace base