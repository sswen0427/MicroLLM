#pragma once

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
                         base::DeviceType device_type);

  static Tensor from_external_cpu(base::DataType data_type,
                                  const std::vector<int32_t>& dims, void* data);

  static Tensor from_external_cuda(base::DataType data_type,
                                   const std::vector<int32_t>& dims,
                                   void* data);

  void to_cpu();

  void to_cuda(cudaStream_t stream = nullptr);

  [[nodiscard]] bool is_empty() const;

  [[nodiscard]] bool is_external() const;

  [[nodiscard]] bool owns_memory() const;

  [[nodiscard]] size_t size() const;

  [[nodiscard]] size_t byte_size() const;

  [[nodiscard]] int32_t dims_size() const;

  [[nodiscard]] base::DataType data_type() const;

  [[nodiscard]] int32_t get_dim(int32_t idx) const;

  [[nodiscard]] base::DeviceType device_type() const;

  [[nodiscard]] tensor::Tensor clone() const;

  void reshape(const std::vector<int32_t>& dims);

  template <typename T>
  [[nodiscard]] T* data();

  template <typename T>
  [[nodiscard]] const T* data() const;

  template <typename T>
  [[nodiscard]] T& at(int64_t offset);

  template <typename T>
  [[nodiscard]] const T& at(int64_t offset) const;

 private:
  static Tensor from_external(base::DataType data_type,
                              const std::vector<int32_t>& dims, void* data,
                              base::DeviceType device_type);

  /**
   * @brief The shape of the tensor (e.g., {Batch, Head, SeqLen, HeadDim}).
   */
  std::vector<int32_t> dims_;

  /**
   * @brief The total number of elements in the tensor (cached for performance).
   */
  std::size_t element_count_ = 0;

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
const T* Tensor::data() const {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null "
         "pointer.";
  return static_cast<const T*>(buffer_->ptr());
}

template <typename T>
T* Tensor::data() {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null "
         "pointer.";
  return static_cast<T*>(buffer_->ptr());
}

template <typename T>
T& Tensor::at(int64_t offset) {
  CHECK(this->device_type() == base::DeviceType::kDeviceCPU)
      << "Fatal: Cannot return CPU reference for a CUDA Tensor!";
  CHECK(offset >= 0 && offset < this->size())
      << "Invalid offset " << offset << " for tensor with size "
      << this->size();
  return *(this->data<T>() + offset);
}

template <typename T>
const T& Tensor::at(int64_t offset) const {
  CHECK(this->device_type() == base::DeviceType::kDeviceCPU)
      << "Fatal: Cannot return CPU reference for a CUDA Tensor!";
  CHECK(offset >= 0 && offset < this->size())
      << "Invalid offset " << offset << " for tensor with size "
      << this->size();
  return *(this->data<T>() + offset);
}

}  // namespace tensor
