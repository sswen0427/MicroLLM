#pragma once

#include <absl/status/statusor.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <safetensors.hh>
#include <string>

#include "model/llama_tensor_names.h"
#include "tensor/tensor.h"

namespace model {

class LlamaSafetensorsLoader {
 public:
  static absl::StatusOr<std::unique_ptr<LlamaSafetensorsLoader>> Open(
      const std::string& safetensors_path);

  absl::StatusOr<tensor::Tensor> LoadTensor(
      const std::string& tensor_name) const;

  absl::StatusOr<tensor::Tensor> LoadGlobalTensor(LlamaTensorKind kind) const;

  absl::StatusOr<tensor::Tensor> LoadLayerTensor(int32_t layer,
                                                 LlamaTensorKind kind) const;

  [[nodiscard]] size_t TensorCount() const;

 private:
  explicit LlamaSafetensorsLoader(
      std::unique_ptr<safetensors::safetensors_t> safetensors);

  std::unique_ptr<safetensors::safetensors_t> safetensors_;
};

}  // namespace model
