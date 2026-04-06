#ifndef MICROLLM_INCLUDE_TENSOR_TENSOR_H
#define MICROLLM_INCLUDE_TENSOR_TENSOR_H
#include <glog/logging.h>

#include <memory>
#include <vector>

#include "base/buffer.h"

namespace tensor {

class Tensor {
 public:
  explicit Tensor() = default;

  explicit Tensor(base::DataType data_type, int32_t dim0);

  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1);

  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,
                  int32_t dim2);

  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,
                  int32_t dim2, int32_t dim3);

  explicit Tensor(base::DataType data_type, std::vector<int32_t> dims);

  template <typename T>
  T* ptr();

  template <typename T>
  const T* ptr() const;

  template <typename T>
  T& index(int64_t offset);

  template <typename T>
  const T& index(int64_t offset) const;

  size_t size() const;

  size_t byte_size() const;

  int32_t dims_size() const;

  base::DataType data_type() const;

  int32_t get_dim(int32_t idx) const;

  void reshape(std::vector<int32_t> dims);

  const std::vector<int32_t>& dims() const;

  std::vector<size_t> strides() const;

  bool assign(std::shared_ptr<base::Buffer> buffer);

  void reset(base::DataType data_type, const std::vector<int32_t>& dims);

  bool allocate(std::shared_ptr<base::DeviceAllocator> allocator,
                bool need_realloc = false);

 private:
  size_t size_ = 0;

  std::vector<int32_t> dims_;

  std::shared_ptr<base::Buffer> buffer_;

  base::DataType data_type_ = base::DataType::kDataTypeUnknown;
};

template <typename T>
T& Tensor::index(int64_t offset) {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());
  T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

template <typename T>
const T& Tensor::index(int64_t offset) const {
  CHECK_GE(offset, 0);
  CHECK_LT(offset, this->size());
  const T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

template <typename T>
const T* Tensor::ptr() const {
  if (!buffer_) {
    return nullptr;
  }
  return const_cast<const T*>(static_cast<T*>(buffer_->ptr()));
}

template <typename T>
T* Tensor::ptr() {
  if (!buffer_) {
    return nullptr;
  }
  return static_cast<T*>(buffer_->ptr());
}
}  // namespace tensor

#endif  // MICROLLM_INCLUDE_TENSOR_TENSOR_H