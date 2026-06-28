#pragma once

#include <cstdint>

#include "tensor/tensor.h"

namespace kernel {

void rope_inplace_kernel_cu(tensor::Tensor &values, int32_t head_count,
                            int32_t head_size, int32_t position,
                            double rope_theta, void *stream = nullptr);

void store_kv_cache_kernel_cu(const tensor::Tensor &key,
                              const tensor::Tensor &value,
                              tensor::Tensor &key_cache,
                              tensor::Tensor &value_cache, int32_t position,
                              int32_t kv_dim, void *stream = nullptr);

void attention_with_cache_kernel_cu(const tensor::Tensor &query,
                                    const tensor::Tensor &key_cache,
                                    const tensor::Tensor &value_cache,
                                    const tensor::Tensor &output,
                                    int32_t position, int32_t head_count,
                                    int32_t head_size, int32_t kv_dim,
                                    int32_t kv_mul, void *stream = nullptr);

}  // namespace kernel
