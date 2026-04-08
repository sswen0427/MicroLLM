
#include "alloc.h"

namespace base {
CUDADeviceAllocator::CUDADeviceAllocator()
    : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void *CUDADeviceAllocator::allocate(std::size_t size) { return nullptr; }

void CUDADeviceAllocator::release(void *ptr) {}

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance =
    nullptr;

}  // namespace base
