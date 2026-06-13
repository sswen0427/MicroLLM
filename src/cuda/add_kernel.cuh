#pragma once

#include "tensor/tensor.h"

namespace kernel {
void add_inplace_kernel_cu(tensor::Tensor &left, const tensor::Tensor &right,
                           void *stream = nullptr);
} // namespace kernel
