#pragma once

#include <boost/noncopyable.hpp>
#include "alloc.h"
#include "base.h"

namespace base {
class Buffer : private boost::noncopyable {
 private:
  std::size_t byte_size_ = 0;

  std::shared_ptr<DeviceAllocator> allocator_;

  void* ptr_ = nullptr;

  bool use_external_ = false;

  DeviceType device_type_ = DeviceType::kDeviceUnknown;

 public:
  explicit Buffer(std::size_t byte_size, DeviceType device_type);

  explicit Buffer(std::size_t byte_size, void* data, DeviceType device_type);

  ~Buffer();

  void copy_from(const Buffer& buffer);

  [[nodiscard]] size_t byte_size() const;

  [[nodiscard]] DeviceType device_type() const;

  [[nodiscard]] bool is_external() const;

  [[nodiscard]] void* ptr() const;
};
}  // namespace base
