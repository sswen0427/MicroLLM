#ifndef MICROLLM_INCLUDE_TENSOR_TENSOR_H
#define MICROLLM_INCLUDE_TENSOR_TENSOR_H
#include <glog/logging.h>

#include <memory>
#include <vector>

#include "base/buffer.h"

namespace tensor {

class Tensor {
 public:
  Tensor() = default;

  static Tensor allocate(base::DataType data_type,
                         const std::vector<int32_t>& dims,
                         const std::shared_ptr<base::DeviceAllocator>& alloc);

  static Tensor from_external(base::DataType data_type,
                              const std::vector<int32_t>& dims, void* ptr);

  void to_cpu();

  void to_cuda(cudaStream_t stream = nullptr);

  [[nodiscard]] bool is_empty() const;

  [[nodiscard]] size_t size() const;

  [[nodiscard]] size_t byte_size() const;

  [[nodiscard]] int32_t dims_size() const;

  [[nodiscard]] std::shared_ptr<base::Buffer> get_buffer() const;

  [[nodiscard]] base::DataType data_type() const;

  [[nodiscard]] int32_t get_dim(int32_t idx) const;

  [[nodiscard]] const std::vector<int32_t>& dims() const;

  [[nodiscard]] base::DeviceType device_type() const;

  [[nodiscard]] tensor::Tensor clone() const;

  void reshape(const std::vector<int32_t>& dims);

  void set_device_type(base::DeviceType device_type);

  template <typename T>
  [[nodiscard]] T* ptr();

  template <typename T>
  [[nodiscard]] const T* ptr() const;

  template <typename T>
  [[nodiscard]] T* ptr(int64_t offset);

  template <typename T>
  [[nodiscard]] const T* ptr(int64_t offset) const;

  template <typename T>
  [[nodiscard]] T& at(int64_t offset);

  template <typename T>
  [[nodiscard]] const T& at(int64_t offset) const;

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

template <typename T>
T& Tensor::at(int64_t offset) {
  CHECK(this->device_type() == base::DeviceType::kDeviceCPU)
      << "Fatal: Cannot return CPU reference for a CUDA Tensor!";
  CHECK(offset >= 0 && offset < this->size())
      << "Invalid offset " << offset << " for tensor with size "
      << this->size();
  return *(this->ptr<T>(offset));
}

template <typename T>
const T& Tensor::at(int64_t offset) const {
  CHECK(this->device_type() == base::DeviceType::kDeviceCPU)
      << "Fatal: Cannot return CPU reference for a CUDA Tensor!";
  CHECK(offset >= 0 && offset < this->size())
      << "Invalid offset " << offset << " for tensor with size "
      << this->size();
  return *(this->ptr<T>(offset));
}

}  // namespace tensor

#endif  // MICROLLM_INCLUDE_TENSOR_TENSOR_H