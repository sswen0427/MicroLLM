#include "cuda/cuda_check.h"
#include "emb_kernel.cuh"

namespace kernel {
__global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num,
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

  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
  }
}

void emb_kernel_cu(const tensor::Tensor &input, const tensor::Tensor &weight,
                   const tensor::Tensor &output, int32_t vocab_size,
                   void *stream) {
  tensor::Tensor input_cu;
  const tensor::Tensor *input_tensor = &input;
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  if (input.device_type() != base::DeviceType::kDeviceCUDA) {
    input_cu = input.clone();
    input_cu.to_cuda(cuda_stream);
    input_tensor = &input_cu;
  }
  const auto input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  CHECK(weight.device_type() == output.device_type());
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  constexpr int32_t thread_num = 128;
  auto *in_ptr = input_tensor->data<int32_t>();
  auto *wei_ptr = const_cast<float *>(weight.data<float>());
  auto *out_ptr = const_cast<float *>(output.data<float>());
  if (stream) {
    emb_kernel_cu_fp32<<<input_num, thread_num, 0, cuda_stream>>>(
        vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
  } else {
    emb_kernel_cu_fp32<<<input_num, thread_num>>>(
        vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
  }
  CHECK_CUDA(cudaGetLastError());
}
}  // namespace kernel
