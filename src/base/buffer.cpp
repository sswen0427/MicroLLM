#include "base/buffer.h"

#include <glog/logging.h>

namespace base {
Buffer::Buffer(const std::size_t byte_size, DeviceType device_type)
    : byte_size_(byte_size), device_type_(device_type) {
  CHECK_GT(byte_size_, 0);
  CHECK(device_type_ != DeviceType::kDeviceUnknown)
      << "Buffer device type must be known.";
  allocator_ = GetDeviceAllocator(device_type_);
  use_external_ = false;
  ptr_ = allocator_->allocate(byte_size_);
}

Buffer::Buffer(const std::size_t byte_size, void* data, DeviceType device_type)
    : byte_size_(byte_size),
      ptr_(data),
      use_external_(true),
      device_type_(device_type) {
  CHECK_GT(byte_size_, 0);
  CHECK(ptr_ != nullptr) << "External buffer pointer must be non-null.";
  CHECK(device_type_ != DeviceType::kDeviceUnknown)
      << "External buffer device type must be known.";
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
    return CopyMemory(ptr_, buffer.ptr(), byte_size, cudaMemcpyHostToHost);
  } else if (buffer_device == DeviceType::kDeviceCUDA &&
             current_device == DeviceType::kDeviceCPU) {
    return CopyMemory(ptr_, buffer.ptr(), byte_size, cudaMemcpyDeviceToHost);
  } else if (buffer_device == DeviceType::kDeviceCPU &&
             current_device == DeviceType::kDeviceCUDA) {
    return CopyMemory(ptr_, buffer.ptr(), byte_size, cudaMemcpyHostToDevice);
  } else {
    return CopyMemory(ptr_, buffer.ptr(), byte_size, cudaMemcpyDeviceToDevice);
  }
}

size_t Buffer::byte_size() const { return byte_size_; }

DeviceType Buffer::device_type() const { return device_type_; }

bool Buffer::is_external() const { return this->use_external_; }

void* Buffer::ptr() const { return ptr_; }

}  // namespace base
