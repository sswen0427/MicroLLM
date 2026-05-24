#include "tensor/tensor.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <limits>

namespace tensor {
namespace {

std::size_t ComputeElementCount(const std::vector<int32_t>& dims) {
  CHECK(!dims.empty()) << "Tensor dims must not be empty.";

  std::size_t size = 1;
  for (const int32_t dim : dims) {
    CHECK_GT(dim, 0) << "Tensor dim must be positive: " << dim;
    CHECK_LE(size, std::numeric_limits<std::size_t>::max() /
                       static_cast<std::size_t>(dim))
        << "Tensor element count overflow.";
    size *= static_cast<std::size_t>(dim);
  }
  return size;
}

std::size_t ComputeByteSize(std::size_t element_count,
                            base::DataType data_type) {
  CHECK(data_type != base::DataType::kDataTypeUnknown)
      << "Tensor data type must be known.";

  const std::size_t type_size = DataTypeSize(data_type);
  CHECK_LE(element_count, std::numeric_limits<std::size_t>::max() / type_size)
      << "Tensor byte size overflow.";
  return element_count * type_size;
}

}  // namespace

Tensor Tensor::allocate(base::DataType data_type,
                        const std::vector<int32_t>& dims,
                        base::DeviceType device_type) {
  Tensor tensor;
  tensor.dims_ = dims;
  tensor.data_type_ = data_type;
  tensor.element_count_ = ComputeElementCount(dims);
  tensor.buffer_ = std::make_shared<base::Buffer>(
      ComputeByteSize(tensor.element_count_, data_type), device_type);
  return tensor;
}

Tensor Tensor::from_external(base::DataType data_type,
                             const std::vector<int32_t>& dims, void* data,
                             base::DeviceType device_type) {
  CHECK_NE(data, nullptr) << "External tensor data pointer must be non-null.";
  CHECK(device_type != base::DeviceType::kDeviceUnknown)
      << "External tensor device type must be known.";

  Tensor tensor;
  tensor.dims_ = dims;
  tensor.data_type_ = data_type;
  tensor.element_count_ = ComputeElementCount(dims);
  tensor.buffer_ = std::make_shared<base::Buffer>(
      ComputeByteSize(tensor.element_count_, data_type), data, device_type);
  return tensor;
}

Tensor Tensor::from_external_cpu(base::DataType data_type,
                                 const std::vector<int32_t>& dims, void* data) {
  return from_external(data_type, dims, data, base::DeviceType::kDeviceCPU);
}

Tensor Tensor::from_external_cuda(base::DataType data_type,
                                  const std::vector<int32_t>& dims,
                                  void* data) {
  return from_external(data_type, dims, data, base::DeviceType::kDeviceCUDA);
}

void Tensor::to_cpu() {
  CHECK_NE(buffer_, nullptr);
  const base::DeviceType& device_type = buffer_->device_type();
  CHECK(device_type != base::DeviceType::kDeviceUnknown)
      << "Unknown device type";

  if (device_type == base::DeviceType::kDeviceCUDA) {
    size_t byte_size = this->byte_size();
    auto cpu_buffer =
        std::make_shared<base::Buffer>(byte_size, base::DeviceType::kDeviceCPU);
    base::CopyMemory(cpu_buffer->ptr(), buffer_->ptr(), byte_size,
                     cudaMemcpyDeviceToHost);
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
    auto cu_buffer = std::make_shared<base::Buffer>(
        byte_size, base::DeviceType::kDeviceCUDA);
    base::CopyMemory(cu_buffer->ptr(), buffer_->ptr(), byte_size,
                     cudaMemcpyHostToDevice, stream);
    this->buffer_ = cu_buffer;
  } else {
    LOG(INFO) << "The device type of the tensor is already cuda.";
  }
}

bool Tensor::is_empty() const {
  return element_count_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
}

bool Tensor::is_external() const {
  return buffer_ != nullptr && buffer_->is_external();
}

bool Tensor::owns_memory() const {
  return buffer_ != nullptr && !buffer_->is_external();
}

size_t Tensor::size() const { return this->element_count_; }

size_t Tensor::byte_size() const {
  if (is_empty()) {
    return 0;
  }
  return ComputeByteSize(element_count_, data_type_);
}

int32_t Tensor::dims_size() const { return static_cast<int32_t>(dims_.size()); }

base::DataType Tensor::data_type() const { return data_type_; }

int32_t Tensor::get_dim(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, this->dims_.size());
  return this->dims_.at(idx);
}
base::DeviceType Tensor::device_type() const {
  CHECK_NE(buffer_, nullptr);
  return buffer_->device_type();
}

void Tensor::reshape(const std::vector<int32_t>& dims) {
  std::size_t new_size = ComputeElementCount(dims);
  CHECK(new_size == element_count_)
      << "Fatal: Reshape cannot change total element count! "
      << "Current size: " << this->element_count_ << ", Requested size: " << new_size;
  this->dims_ = dims;
}

Tensor Tensor::clone() const {
  CHECK_NE(buffer_, nullptr);

  Tensor new_tensor;
  new_tensor.dims_ = this->dims_;
  new_tensor.data_type_ = this->data_type_;
  new_tensor.element_count_ = this->element_count_;

  new_tensor.buffer_ =
      std::make_shared<base::Buffer>(this->byte_size(), buffer_->device_type());
  new_tensor.buffer_->copy_from(*buffer_.get());
  return new_tensor;
}

}  // namespace tensor
