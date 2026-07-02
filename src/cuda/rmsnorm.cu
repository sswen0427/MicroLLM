#include <device_launch_parameters.h>

#include <cub/block/block_reduce.cuh>

#include "cuda/cuda_check.h"
#include "rmsnorm.cuh"
namespace kernel {
namespace {

/**
 * @brief Applies RMSNorm to one input vector.
 *
 * Tensor shapes in logical layout:
 *   in:  [size]
 *   wei: [size]
 *   out: [size]
 *
 * A single block handles the whole vector. Threads first compute partial sums
 * of x^2, CUB reduces them to the full sum, then each thread writes its part
 * of: out[i] = in[i] * wei[i] / sqrt(mean(in^2) + eps)
 */
template <int32_t BLOCK_DIM>
__global__ void RmsNormKernel(const float *in, const float *wei, float *out,
                              int size, float eps) {
  const int tid = threadIdx.x;

  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  const float4 *in_pack = reinterpret_cast<const float4 *>(in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += in[i] * in[i];
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  const float4 *wei_pack = reinterpret_cast<const float4 *>(wei);
  float4 *out_pack = reinterpret_cast<float4 *>(out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) = make_float4(
        scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
        scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    out[i] = wei[i] * in[i] * scale;
  }
}

}  // namespace

void RmsNormCuda(const tensor::Tensor &input, const tensor::Tensor &weight,
                 const tensor::Tensor &output, void *stream, float eps) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(input.data_type() == base::DataType::kDataTypeFp32);
  CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
  CHECK(output.data_type() == base::DataType::kDataTypeFp32);
  CHECK_EQ(input.dims_size(), 1);
  CHECK_EQ(weight.dims_size(), 1);
  CHECK_EQ(output.dims_size(), 1);
  CHECK_EQ(input.size(), weight.size());
  CHECK_EQ(input.size(), output.size());

  const int32_t size = static_cast<int32_t>(input.size());
  const float *in_ptr = input.data<float>();
  const float *wei_ptr = weight.data<float>();
  float *out_ptr = const_cast<float *>(output.data<float>());
  CHECK_EQ(reinterpret_cast<std::uintptr_t>(in_ptr) % alignof(float4), 0);
  CHECK_EQ(reinterpret_cast<std::uintptr_t>(wei_ptr) % alignof(float4), 0);
  CHECK_EQ(reinterpret_cast<std::uintptr_t>(out_ptr) % alignof(float4), 0);
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    RmsNormKernel<128>
        <<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  } else {
    RmsNormKernel<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  }
  CHECK_CUDA(cudaGetLastError());
}
}  // namespace kernel
