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

  std::shared_ptr<DeviceAllocator> allocator_;

  void* ptr_ = nullptr;

  bool use_external_ = false;

  DeviceType device_type_ = DeviceType::kDeviceUnknown;

 public:
  explicit Buffer() = default;

  /**
   * @brief If the user provides a ptr, then the buffer will be external,
   * otherwise it will be internal, and use the allocator to allocate memory.
   */
  explicit Buffer(std::size_t byte_size,
                  std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr);

  virtual ~Buffer();

  void copy_from(const Buffer& buffer);

  [[nodiscard]] std::shared_ptr<DeviceAllocator> allocator() const;

  [[nodiscard]] size_t byte_size() const;

  [[nodiscard]] DeviceType device_type() const;

  [[nodiscard]] bool is_external() const;

  [[nodiscard]] void* ptr() const;

  void set_device_type(DeviceType device_type);
};
}  // namespace base

#endif
