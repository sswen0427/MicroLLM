#pragma once

#include "tensor/tensor.h"

namespace kernel {
void matmul_kernel_cu(const tensor::Tensor &input, const tensor::Tensor &weight,
                      const tensor::Tensor &output, float scale = 1.f,
                      void *stream = nullptr);
} // namespace kernel
