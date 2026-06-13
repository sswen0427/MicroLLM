#pragma once

#include "base/types.h"
#include "tensor/tensor.h"
namespace kernel {
typedef void (*MatmulKernel)(const tensor::Tensor& input,
                             const tensor::Tensor& weight,
                             const tensor::Tensor& output, float scale,
                             const base::CudaConfig* config);

typedef void (*EmbeddingKernel)(const tensor::Tensor& input,
                                const tensor::Tensor& weight,
                                const tensor::Tensor& output,
                                int32_t vocab_size, void* stream);

typedef void (*SwigluKernel)(const tensor::Tensor& input1,
                             const tensor::Tensor& input2,
                             const tensor::Tensor& output, void* stream);

typedef void (*RMSNormKernel)(const tensor::Tensor& input,
                              const tensor::Tensor& weight,
                              const tensor::Tensor& output, void* stream);

EmbeddingKernel get_emb_kernel(base::DeviceType device_type);

MatmulKernel get_matmul_kernel(base::DeviceType device_type);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

SwigluKernel get_swiglu_kernel(base::DeviceType device_type,
                               void* stream = nullptr);
}  // namespace kernel
