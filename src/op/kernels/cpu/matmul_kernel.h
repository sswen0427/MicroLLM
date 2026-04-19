#pragma once

#include "base/base.h"
#include "tensor/tensor.h"
namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input,
                       const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale = 1.f,
                       const base::CudaConfig* config = nullptr);
}  // namespace kernel
