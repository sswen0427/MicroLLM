#include <cub/block/block_reduce.cuh>

#include "cuda/cuda_check.h"
#include "matmul.cuh"
#include "tensor/tensor.h"

namespace kernel {
namespace {

/**
 * @brief Computes output[batch, row] = dot(input[batch], weight[row]).
 *
 * Tensor shapes in logical row-major layout:
 *   input:  [batch, M]
 *   weight: [K, M]
 *   output: [batch, K]
 *
 * Single-token GEMV uses this same kernel with batch = 1.
 */
template <int THREAD_PER_BLOCK>
__global__ void MatmulBatchKernel(const float *input, const float *weight,
                                  float *output, int batch, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int batch_idx = blockIdx.y;
  if (row >= K || batch_idx >= batch) {
    return;
  }

  const float *input_row = input + batch_idx * M;
  const float *weight_row = weight + row * M;

  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_size * pack_num;
  // Only use float4 when every row starts at a 16-byte aligned offset. If M is
  // not divisible by 4, later rows start at unaligned addresses, so use the
  // scalar path for the whole row instead of mixing vector and tail loads.
  const bool use_float4 = (M % pack_size) == 0;

  float sum = 0.0f;
  if (use_float4) {
    const float4 *input_float4_ptr =
        reinterpret_cast<const float4 *>(input_row);
    const float4 *weight_float4_ptr =
        reinterpret_cast<const float4 *>(weight_row);
    for (int i = tid; i < pack_num; i += blockDim.x) {
      const float4 input_float4 = *(input_float4_ptr + i);
      const float4 weight_float4 = *(weight_float4_ptr + i);
      sum +=
          input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
          input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
    }
  }

  const int scalar_start = use_float4 ? pack_off : 0;
  for (int col = scalar_start + tid; col < M; col += blockDim.x) {
    sum += input_row[col] * weight_row[col];
  }
  sdata[tid] = sum;
  __syncthreads();

  using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage temp;
  sum = BlockReduce(temp).Sum(sdata[tid]);
  if (tid == 0) {
    output[batch_idx * K + row] = sum;
  }
}

void LaunchMatmulBatch(const tensor::Tensor &input, const tensor::Tensor &weight,
                       const tensor::Tensor &output, int32_t batch, int32_t M,
                       int32_t K, void *stream) {
  constexpr int threads = 128;
  const dim3 grid(K, batch);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  MatmulBatchKernel<threads><<<grid, threads, 0, cuda_stream>>>(
      input.data<float>(), weight.data<float>(),
      const_cast<float *>(output.data<float>()), batch, M, K);
  CHECK_CUDA(cudaGetLastError());
}

}  // namespace

void MatmulCuda(const tensor::Tensor &input, const tensor::Tensor &weight,
                const tensor::Tensor &output, const float scale, void *stream) {
  (void)scale;
  CHECK(!input.is_empty());
  CHECK_EQ(input.dims_size(), 1);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(input.data_type() == base::DataType::kDataTypeFp32);

  CHECK(!weight.is_empty());
  CHECK_EQ(weight.dims_size(), 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(weight.data_type() == base::DataType::kDataTypeFp32);

  CHECK(!output.is_empty());
  CHECK_EQ(output.dims_size(), 1);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(output.data_type() == base::DataType::kDataTypeFp32);

  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col

  CHECK_EQ(M, input.get_dim(0));
  CHECK_EQ(K, output.get_dim(0));
  constexpr int32_t batch = 1;
  LaunchMatmulBatch(input, weight, output, batch, M, K, stream);
}

void MatmulBatchCuda(const tensor::Tensor &input, const tensor::Tensor &weight,
                     const tensor::Tensor &output, const float scale,
                     void *stream) {
  (void)scale;
  CHECK(!input.is_empty());
  CHECK_EQ(input.dims_size(), 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(input.data_type() == base::DataType::kDataTypeFp32);

  CHECK(!weight.is_empty());
  CHECK_EQ(weight.dims_size(), 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(weight.data_type() == base::DataType::kDataTypeFp32);

  CHECK(!output.is_empty());
  CHECK_EQ(output.dims_size(), 2);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(output.data_type() == base::DataType::kDataTypeFp32);

  const int32_t batch = input.get_dim(0);
  const int32_t M = input.get_dim(1);
  const int32_t K = weight.get_dim(0);
  CHECK_EQ(weight.get_dim(1), M);
  CHECK_EQ(output.get_dim(0), batch);
  CHECK_EQ(output.get_dim(1), K);

  LaunchMatmulBatch(input, weight, output, batch, M, K, stream);
}
}  // namespace kernel
