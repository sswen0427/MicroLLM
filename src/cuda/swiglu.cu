#include "cuda/cuda_check.h"
#include "swiglu.cuh"
#include "tensor/tensor.h"

namespace kernel {
namespace {

__global__ void SwiGluKernel(int size, const float *in1, const float *in2,
                             float *out) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }

  const float gate = in1[idx];
  const float silu = gate / (1.0f + expf(-gate));
  out[idx] = silu * in2[idx];
}

}  // namespace

void SwiGluCuda(const tensor::Tensor &input1, const tensor::Tensor &input2,
                const tensor::Tensor &output, void *stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(input2.is_empty(), false);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(output.is_empty(), false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  int size = static_cast<int32_t>(input1.size());
  int threads = 128;
  int blocks = (size + threads - 1) / threads;
  if (!stream) {
    SwiGluKernel<<<blocks, threads>>>(
        size, input1.data<float>(), input2.data<float>(),
        const_cast<float *>(output.data<float>()));
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    SwiGluKernel<<<blocks, threads, 0, stream_>>>(
        size, input1.data<float>(), input2.data<float>(),
        const_cast<float *>(output.data<float>()));
  }
  CHECK_CUDA(cudaGetLastError());
}
}  // namespace kernel
