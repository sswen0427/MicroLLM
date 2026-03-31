#include "tensor.h"

#include <cstddef>
#include <numeric>

#include "glog/logging.h"

template <typename T, typename Tp>
static inline size_t MutiplyAccumulate(T begin, T end, Tp init) {
  size_t size = std::accumulate(begin, end, init);
  return size;
}

Tensor::Tensor(DataType data_type, int32_t dim0) : data_type_(data_type) {
  dims_.push_back(dim0);
  size_ = dim0;
}

Tensor::Tensor(DataType data_type, int32_t dim0, int32_t dim1)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  size_ = dim0 * dim1;
}

Tensor::Tensor(DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  size_ = dim0 * dim1 * dim2;
}

Tensor::Tensor(DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
               int32_t dim3)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  dims_.push_back(dim3);
  size_ = dim0 * dim1 * dim2 * dim3;
}

Tensor::Tensor(DataType data_type, std::vector<int32_t> dims)
    : dims_(std::move(dims)), data_type_(data_type) {
  size_ = MutiplyAccumulate(dims_.begin(), dims_.end(), 1);
}

size_t Tensor::size() const { return size_; }

int32_t Tensor::get_dim(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, dims_.size());
  return dims_.at(idx);
}

bool Tensor::assign(std::shared_ptr<Buffer> buffer) {
  if (!buffer) {
    LOG(ERROR) << "Tensor::assign: buffer is null";
    return false;
  }
  size_t byte_size = this->byte_size();
  if (byte_size != buffer->byte_size()) {
    LOG(ERROR) << "Tensor::assign: byte_size not match";
    return false;
  }
  buffer_ = buffer;
  return true;
}

bool Tensor::allocate(std::shared_ptr<DeviceAllocator> allocator,
                      bool need_realloc) {
  if (!allocator) {
    LOG(ERROR) << "Tensor::allocate: allocator is null";
    return false;
  }
  size_t byte_size = this->byte_size();
  if (!byte_size) {
    LOG(ERROR) << "Tensor::allocate: byte_size is 0";
    return false;
  }

  if (buffer_ && byte_size == buffer_->byte_size()) {
    if (!need_realloc) {
      return true;
    }
  }
  buffer_ = std::make_shared<Buffer>(byte_size, allocator, nullptr);
  if (!buffer_->ptr()) {
    LOG(ERROR) << "Tensor::allocate: failed to allocate buffer";
    return false;
  }
  return true;
}

const std::vector<int32_t>& Tensor::dims() const { return dims_; }

void Tensor::reset(DataType data_type, const std::vector<int32_t>& dims) {
  data_type_ = data_type;
  dims_ = dims;
  size_ = MutiplyAccumulate(dims_.begin(), dims_.end(), 1);
  buffer_ = nullptr;
}

int32_t Tensor::dims_size() const { return static_cast<int32_t>(dims_.size()); }

DataType Tensor::data_type() const { return data_type_; }

void Tensor::reshape(std::vector<int32_t> dims) {
  size_t size = MutiplyAccumulate(dims.begin(), dims.end(), 1);
  if (buffer_ != nullptr && size == size_) {
    dims_ = dims;
  } else {
    dims_ = dims;
    size_ = size;
    buffer_ = std::make_shared<Buffer>(size, buffer_->allocator());
    CHECK(buffer_->allocate());
  }
}

size_t Tensor::byte_size() const { return size() * DataTypeSize(data_type_); }

std::vector<size_t> Tensor::strides() const {
  std::vector<size_t> strides;
  if (!dims_.empty()) {
    for (int32_t i = 0; i < dims_.size() - 1; ++i) {
      size_t stride = MutiplyAccumulate(dims_.begin(), dims_.end(), 1);
      strides.push_back(stride);
    }
    strides.push_back(1);
  }
  return strides;
}
