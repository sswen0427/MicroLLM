#pragma once

#include <tensor/tensor.h>
namespace kernel {
void scale_inplace_cpu(float scale, const tensor::Tensor& tensor,
                       void* stream = nullptr);
}