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

  explicit Tensor(base::DataType data_type, std::vector<int32_t> dims,
                  bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                  void* ptr = nullptr);

  void to_cpu();

  void to_cuda(cudaStream_t stream = nullptr);

  bool is_empty() const;

  void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc,
                   base::DataType data_type, bool need_alloc, void* ptr);

  template <typename T>
  T* ptr();

  template <typename T>
  const T* ptr() const;

  void reshape(const std::vector<int32_t>& dims);

  std::shared_ptr<base::Buffer> get_buffer() const;

  size_t size() const;

  size_t byte_size() const;

  int32_t dims_size() const;

  base::DataType data_type() const;

  int32_t get_dim(int32_t idx) const;

  const std::vector<int32_t>& dims() const;

  std::vector<size_t> strides() const;

  bool assign(std::shared_ptr<base::Buffer> buffer);

  void reset(base::DataType data_type, const std::vector<int32_t>& dims);

  void set_device_type(base::DeviceType device_type);

  base::DeviceType device_type() const;

  bool allocate(std::shared_ptr<base::DeviceAllocator> allocator,
                bool need_realloc = false);

  template <typename T>
  T* ptr(int64_t index);

  template <typename T>
  const T* ptr(int64_t index) const;

  template <typename T>
  T& index(int64_t offset);

  template <typename T>
  const T& index(int64_t offset) const;

  tensor::Tensor clone() const;

 private:
  /**
   * @brief The shape of the tensor (e.g., {Batch, Head, SeqLen, HeadDim}).
   */
  std::vector<int32_t> dims_;

  /**
   * @brief The total number of elements in the tensor (cached for performance).
   */
  std::size_t size_ = 0;

  /**
   * @brief The data type of each element (e.g., FP32, INT8).
   */
  base::DataType data_type_ = base::DataType::kDataTypeUnknown;

  /**
   * @brief The underlying physical memory buffer (CPU or GPU) that holds the
   * actual data.
   */
  std::shared_ptr<base::Buffer> buffer_;
};

template <typename T>
T& Tensor::index(int64_t offset) {
  CHECK(this->device_type() == base::DeviceType::kDeviceCPU)
      << "Fatal: Cannot return CPU reference for a CUDA Tensor!";
  CHECK(offset >= 0 && offset < this->size())
      << "Invalid offset " << offset << " for tensor with size "
      << this->size();
  return *(this->ptr<T>(offset));
}

template <typename T>
const T& Tensor::index(int64_t offset) const {
  CHECK(this->device_type() == base::DeviceType::kDeviceCPU)
      << "Fatal: Cannot return CPU reference for a CUDA Tensor!";
  CHECK(offset >= 0 && offset < this->size())
      << "Invalid offset " << offset << " for tensor with size "
      << this->size();
  return *(this->ptr<T>(offset));
}

template <typename T>
const T* Tensor::ptr() const {
  if (!buffer_) {
    return nullptr;
  }
  return static_cast<const T*>(buffer_->ptr());
}

template <typename T>
T* Tensor::ptr() {
  if (!buffer_) {
    return nullptr;
  }
  return static_cast<T*>(buffer_->ptr());
}

template <typename T>
T* Tensor::ptr(int64_t index) {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null "
         "pointer.";
  return this->ptr<T>() + index;
}

template <typename T>
const T* Tensor::ptr(int64_t index) const {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null "
         "pointer.";
  return this->ptr<T>() + index;
}
}  // namespace tensor

#endif  // MICROLLM_INCLUDE_TENSOR_TENSOR_H