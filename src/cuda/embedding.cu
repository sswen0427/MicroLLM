#include "cuda/cuda_check.h"
#include "embedding.cuh"

namespace kernel {
namespace {

/**
 * @brief Embedding kernel.
 *
 * Tensor shapes in logical row-major layout:
 *   input_ptr:  [token_num]
 *   weight_ptr: [vocab_size, weight_dim]
 *   output_ptr: [token_num, weight_dim]
 *
 * Copies token embeddings from the weight table to the output tensor. Each
 * block handles one token, and threads within the block cooperate to copy that
 * token's embedding vector:
 *   output_ptr[token_idx, dim] = weight_ptr[input_ptr[token_idx], dim]
 */
__global__ void EmbeddingKernel(int32_t vocab_size, int32_t token_num,
                                int32_t weight_dim, const int32_t *input_ptr,
                                const float *weight_ptr, float *output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) {
    return;
  }
  int32_t token = input_ptr[token_idx];
  if (token < 0 || token >= vocab_size) {
    return;
  }

  float *output_ptr_start = output_ptr + token_idx * weight_dim;
  const float *weight_ptr_start = weight_ptr + token * weight_dim;

  // Adjacent threads copy adjacent elements each round for coalesced access.
  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
  }
}

}  // namespace

void EmbeddingCuda(const tensor::Tensor &input, const tensor::Tensor &weight,
                   const tensor::Tensor &output, int32_t vocab_size,
                   void *stream) {
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  const auto input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  constexpr int32_t thread_num = 128;
  auto *in_ptr = input.data<int32_t>();
  auto *wei_ptr = const_cast<float *>(weight.data<float>());
  auto *out_ptr = const_cast<float *>(output.data<float>());
  EmbeddingKernel<<<input_num, thread_num, 0, cuda_stream>>>(
      vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
  CHECK_CUDA(cudaGetLastError());
}
}  // namespace kernel
