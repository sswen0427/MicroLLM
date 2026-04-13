#include "buffer.h"

#include <glog/logging.h>

namespace base {
Buffer::Buffer(const std::size_t byte_size,
               std::shared_ptr<DeviceAllocator> allocator, void* ptr)
    : byte_size_(byte_size), allocator_(std::move(allocator)), ptr_(ptr) {
  CHECK(bool(allocator_) != bool(ptr_))
      << "The two pointers must be different.";
  if (allocator_) {
    device_type_ = allocator_->device_type();
    use_external_ = false;
    ptr_ = allocator_->allocate(byte_size);
  } else {
    device_type_ = DeviceType::kDeviceCPU;
    use_external_ = true;
  }
}

Buffer::~Buffer() {
  if (!use_external_) {
    CHECK(allocator_ && ptr_) << "The pointer must be non-null.";
    allocator_->release(ptr_);
    ptr_ = nullptr;
  }
}

void Buffer::copy_from(const Buffer& buffer) {
  CHECK(allocator_ != nullptr) << "The allocator pointer must be non-null.";
  CHECK(buffer.ptr_ != nullptr) << "The buffer pointer must be non-null.";
  CHECK(byte_size_ >= buffer.byte_size_)
      << "The dst byte size " << byte_size_
      << " must be greater than or equal to the src byte size "
      << buffer.byte_size_;

  size_t byte_size = buffer.byte_size_;
  const DeviceType& buffer_device = buffer.device_type();
  const DeviceType& current_device = this->device_type();
  CHECK(buffer_device != DeviceType::kDeviceUnknown &&
        current_device != DeviceType::kDeviceUnknown)
      << "The device type must be known.";

  if (buffer_device == DeviceType::kDeviceCPU &&
      current_device == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(ptr_, buffer.ptr(), byte_size,
                              cudaMemcpyHostToHost, nullptr);
  } else if (buffer_device == DeviceType::kDeviceCUDA &&
             current_device == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(ptr_, buffer.ptr(), byte_size,
                              cudaMemcpyDeviceToHost, nullptr);
  } else if (buffer_device == DeviceType::kDeviceCPU &&
             current_device == DeviceType::kDeviceCUDA) {
    return allocator_->memcpy(ptr_, buffer.ptr(), byte_size,
                              cudaMemcpyHostToDevice, nullptr);
  } else {
    return allocator_->memcpy(ptr_, buffer.ptr(), byte_size,
                              cudaMemcpyDeviceToDevice, nullptr);
  }
}

std::shared_ptr<DeviceAllocator> Buffer::allocator() const {
  return allocator_;
}

size_t Buffer::byte_size() const { return byte_size_; }

DeviceType Buffer::device_type() const { return device_type_; }

bool Buffer::is_external() const { return this->use_external_; }

void* Buffer::ptr() const { return ptr_; }

void Buffer::set_device_type(DeviceType device_type) {
  device_type_ = device_type;
}

}  // namespace base