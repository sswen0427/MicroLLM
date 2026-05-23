#pragma once

#include <absl/status/statusor.h>

#include <safetensors.hh>
#include <string>

#include "tensor/tensor_view.h"

namespace model {

absl::StatusOr<tensor::TensorView> GetSafetensorsTensorView(
    const safetensors::safetensors_t& safetensors,
    const std::string& tensor_name);

}  // namespace model
