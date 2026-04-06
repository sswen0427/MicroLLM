#ifndef MICROLLM_INCLUDE_BASE_BUFFER_H
#define MICROLLM_INCLUDE_BASE_BUFFER_H
#include <memory>

#include "alloc.h"
#include "base.h"

namespace base {
class Buffer : public Noncopyable {
 public:
  explicit Buffer() = default;

  explicit Buffer(std::size_t byte_size,
                  std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr, bool use_external = false);

  virtual ~Buffer();

  bool allocate();

  void* ptr();

  const void* ptr() const;

  size_t byte_size() const;

  std::shared_ptr<DeviceAllocator> allocator() const;

 private:
  std::size_t byte_size_ = 0;

  void* ptr_ = nullptr;

  bool use_external_ = false;

  std::shared_ptr<DeviceAllocator> allocator_;
};
}  // namespace base

#endif
