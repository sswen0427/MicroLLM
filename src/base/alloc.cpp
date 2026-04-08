#include "base/alloc.h"

namespace base {
void DeviceAllocator::memcpy(void *dst, const void *src, std::size_t size,
                             MemcpyType type, void *stream, bool need_sync) {}

void DeviceAllocator::memset_zero(void *ptr, std::size_t size) {}

}  // namespace base
