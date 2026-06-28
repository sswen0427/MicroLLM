#pragma once

#include "tensor/tensor.h"

namespace kernel {
void RmsNormCuda(const tensor::Tensor &input, const tensor::Tensor &weight,
                 const tensor::Tensor &output, void *stream = nullptr,
                 float eps = 1e-5f);
}  // namespace kernel
