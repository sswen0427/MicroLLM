#include "tensor/tensor.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <numeric>

namespace tensor {

Tensor Tensor::allocate(base::DataType data_type,
                        const std::vector<int32_t>& dims,
                        const std::shared_ptr<base::DeviceAllocator>& alloc) {
  Tensor tensor;
  tensor.dims_ = dims;
  tensor.data_type_ = data_type;
  tensor.size_ =
      std::accumulate(dims.begin(), dims.end(), 1LL, std::multiplies<>());
  tensor.buffer_ = std::make_shared<base::Buffer>(
      tensor.size_ * DataTypeSize(data_type), alloc, nullptr);
  return tensor;
}

Tensor Tensor::from_external(base::DataType data_type,
                             const std::vector<int32_t>& dims, void* ptr) {
  Tensor tensor;
  tensor.dims_ = dims;
  tensor.data_type_ = data_type;
  tensor.size_ =
      std::accumulate(dims.begin(), dims.end(), 1LL, std::multiplies<>());
  tensor.buffer_ = std::make_shared<base::Buffer>(
      tensor.size_ * DataTypeSize(data_type), nullptr, ptr);
  return tensor;
}

void Tensor::to_cpu() {
  CHECK_NE(buffer_, nullptr);
  const base::DeviceType& device_type = buffer_->device_type();
  CHECK(device_type != base::DeviceType::kDeviceUnknown)
      << "Unknown device type";

  if (device_type == base::DeviceType::kDeviceCUDA) {
    size_t byte_size = this->byte_size();
    auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
    auto cpu_buffer = std::make_shared<base::Buffer>(byte_size, cpu_alloc);
    cpu_alloc->memcpy(cpu_buffer->ptr(), buffer_->ptr(), byte_size,
                      cudaMemcpyDeviceToHost, nullptr);
    this->buffer_ = cpu_buffer;
  } else {
    LOG(INFO) << "The device type of the tensor is already cpu.";
  }
}

void Tensor::to_cuda(cudaStream_t stream) {
  CHECK_NE(buffer_, nullptr);
  const base::DeviceType device_type = this->device_type();
  CHECK(device_type != base::DeviceType::kDeviceUnknown)
      << "Unknown device type";
  if (device_type == base::DeviceType::kDeviceCPU) {
    size_t byte_size = this->byte_size();
    auto cu_alloc = base::CUDADeviceAllocatorFactory::get_instance();
    auto cu_buffer = std::make_shared<base::Buffer>(byte_size, cu_alloc);
    cu_alloc->memcpy(cu_buffer->ptr(), buffer_->ptr(), byte_size,
                     cudaMemcpyHostToDevice, stream);
    this->buffer_ = cu_buffer;
  } else {
    LOG(INFO) << "The device type of the tensor is already cuda.";
  }
}

bool Tensor::is_empty() const {
  return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
}

size_t Tensor::size() const { return this->size_; }

size_t Tensor::byte_size() const {
  return this->size() * DataTypeSize(data_type_);
}

int32_t Tensor::dims_size() const { return static_cast<int32_t>(dims_.size()); }

std::shared_ptr<base::Buffer> Tensor::get_buffer() const { return buffer_; }

base::DataType Tensor::data_type() const { return data_type_; }

int32_t Tensor::get_dim(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, this->dims_.size());
  return this->dims_.at(idx);
}
const std::vector<int32_t>& Tensor::dims() const { return this->dims_; }

base::DeviceType Tensor::device_type() const { return buffer_->device_type(); }

void Tensor::set_device_type(base::DeviceType device_type) {
  buffer_->set_device_type(device_type);
}

void Tensor::reshape(const std::vector<int32_t>& dims) {
  std::size_t new_size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
  CHECK(new_size == size_)
      << "Fatal: Reshape cannot change total element count! "
      << "Current size: " << this->size_ << ", Requested size: " << new_size;
  this->dims_ = dims;
}

Tensor Tensor::clone() const {
  Tensor new_tensor;
  new_tensor.dims_ = this->dims_;
  new_tensor.data_type_ = this->data_type_;
  new_tensor.size_ = this->size_;

  auto allocator = buffer_->allocator();
  new_tensor.buffer_ =
      std::make_shared<base::Buffer>(this->byte_size(), allocator);
  new_tensor.buffer_->copy_from(*buffer_.get());
  return new_tensor;
}

}  // namespace tensor