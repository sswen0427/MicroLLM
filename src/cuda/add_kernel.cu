#include "add_kernel.cuh"

#include "cuda/cuda_check.h"

namespace kernel {
namespace {

__global__ void add_inplace_kernel_cu_fp32(float *left, const float *right,
                                           int size) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }
  left[idx] += right[idx];
}

} // namespace

void add_inplace_kernel_cu(tensor::Tensor &left, const tensor::Tensor &right,
                           void *stream) {
  CHECK(!left.is_empty());
  CHECK(!right.is_empty());
  CHECK_EQ(left.size(), right.size());
  CHECK(left.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(right.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(left.data_type() == base::DataType::kDataTypeFp32);
  CHECK(right.data_type() == base::DataType::kDataTypeFp32);

  constexpr int threads = 128;
  const int size = static_cast<int>(left.size());
  const int blocks = (size + threads - 1) / threads;
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  if (cuda_stream != nullptr) {
    add_inplace_kernel_cu_fp32<<<blocks, threads, 0, cuda_stream>>>(
        left.data<float>(), right.data<float>(), size);
  } else {
    add_inplace_kernel_cu_fp32<<<blocks, threads>>>(left.data<float>(),
                                                    right.data<float>(), size);
  }
  CHECK_CUDA(cudaGetLastError());
}

} // namespace kernel
