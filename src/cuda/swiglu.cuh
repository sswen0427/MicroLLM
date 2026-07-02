#pragma once

#include "tensor/tensor.h"

namespace kernel {
void SwiGluCuda(const tensor::Tensor& gate, const tensor::Tensor& up,
                const tensor::Tensor& output, void* stream = nullptr);
}  // namespace kernel
