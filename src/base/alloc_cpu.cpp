#include "base/alloc.h"
#include "base/base.h"

namespace base {
CPUDeviceAllocator::CPUDeviceAllocator()
    : DeviceAllocator(DeviceType::kDeviceCPU) {}

void *CPUDeviceAllocator::allocate(std::size_t size) { return nullptr; }

void CPUDeviceAllocator::release(void *ptr) {}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance =
    nullptr;
}  // namespace base