#pragma once

#include <cstddef>

#include "tensor/tensor.h"

namespace model {

float TensorElementAsFloat(const tensor::Tensor& tensor, size_t offset);

}  // namespace model
