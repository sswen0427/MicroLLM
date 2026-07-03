#include "cuda/cuda_check.h"
#include "swiglu.cuh"
#include "tensor/tensor.h"

namespace kernel {
namespace {

/**
 * @brief Applies the SwiGLU activation used in the LLaMA feed-forward network.
 *
 *   gate = W_gate * x
 *   up   = W_up   * x
 *   out  = SiLU(gate) * up
 *        = gate / (1 + exp(-gate)) * up
 *
 * This kernel assumes gate and up already have the same 1D shape and computes
 * the element-wise activation output.
 */
__global__ void SwiGluKernel(int size, const float* gate, const float* up,
                             float* out) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }

  const float gate_value = gate[idx];
  const float silu = gate_value / (1.0f + expf(-gate_value));
  out[idx] = silu * up[idx];
}

}  // namespace

void SwiGluCuda(const tensor::Tensor& gate, const tensor::Tensor& up,
                const tensor::Tensor& output, void* stream) {
  CHECK(!gate.is_empty());
  CHECK(!up.is_empty());
  CHECK(!output.is_empty());

  CHECK(gate.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(up.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(gate.data_type() == base::DataType::kDataTypeFp32);
  CHECK(up.data_type() == base::DataType::kDataTypeFp32);
  CHECK(output.data_type() == base::DataType::kDataTypeFp32);

  CHECK_EQ(gate.size(), up.size());
  CHECK_EQ(gate.size(), output.size());

  int size = static_cast<int32_t>(gate.size());
  int threads = 128;
  int blocks = (size + threads - 1) / threads;
  if (!stream) {
    SwiGluKernel<<<blocks, threads>>>(size, gate.data<float>(),
                                      up.data<float>(),
                                      const_cast<float*>(output.data<float>()));
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    SwiGluKernel<<<blocks, threads, 0, stream_>>>(
        size, gate.data<float>(), up.data<float>(),
        const_cast<float*>(output.data<float>()));
  }
  CHECK_CUDA(cudaGetLastError());
}
}  // namespace kernel
