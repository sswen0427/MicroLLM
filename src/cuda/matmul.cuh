#pragma once

#include "tensor/tensor.h"

namespace kernel {
void MatmulCuda(const tensor::Tensor &input, const tensor::Tensor &weight,
                const tensor::Tensor &output, float scale = 1.f,
                void *stream = nullptr);

void MatmulBatchCuda(const tensor::Tensor &input, const tensor::Tensor &weight,
                     const tensor::Tensor &output, float scale = 1.f,
                     void *stream = nullptr);
}  // namespace kernel
