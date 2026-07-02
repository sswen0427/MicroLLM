#pragma once

#include "tensor/tensor.h"

namespace kernel {
void AddInPlaceCuda(tensor::Tensor &left, const tensor::Tensor &right,
                    void *stream = nullptr);
}  // namespace kernel
