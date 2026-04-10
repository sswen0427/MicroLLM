#ifndef MICROLLM_INCLUDE_BASE_BUFFER_H
#define MICROLLM_INCLUDE_BASE_BUFFER_H
#include <boost/noncopyable.hpp>
#include <memory>

#include "alloc.h"
#include "base.h"

namespace base {
class Buffer : private boost::noncopyable {
 private:
  std::size_t byte_size_ = 0;

  void* ptr_ = nullptr;

  bool use_external_ = false;

  DeviceType device_type_ = DeviceType::kDeviceUnknown;

  std::shared_ptr<DeviceAllocator> allocator_;

 public:
  explicit Buffer() = default;

  explicit Buffer(std::size_t byte_size,
                  std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr, bool use_external = false);

  virtual ~Buffer();

  bool allocate();

  void copy_from(const Buffer& buffer) const;

  void* ptr();

  const void* ptr() const;

  size_t byte_size() const;

  std::shared_ptr<DeviceAllocator> allocator() const;

  DeviceType device_type() const;

  void set_device_type(DeviceType device_type);

  bool is_external() const;
};
}  // namespace base

#endif
