#include <device_launch_parameters.h>

#include <cub/block/block_reduce.cuh>

#include "cuda/cuda_check.h"
#include "rmsnorm.cuh"
namespace kernel {
namespace {

template <int32_t BLOCK_DIM>
__global__ void RmsNormBatchKernel(const float *in, const float *wei,
                                   float *out, int batch, int hidden_size,
                                   float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= batch) {
    return;
  }

  const float *in_row = in + row * hidden_size;
  float *out_row = out + row * hidden_size;

  constexpr int pack_size = 4;
  const int pack_num = hidden_size / pack_size;
  const int pack_off = pack_size * pack_num;
  const bool use_float4 = (hidden_size % pack_size) == 0;

  float sum = 0.0f;
  if (use_float4) {
    const float4 *in_pack = reinterpret_cast<const float4 *>(in_row);
    for (int i = tid; i < pack_num; i += blockDim.x) {
      const float4 in_float4 = *(in_pack + i);
      sum += in_float4.x * in_float4.x;
      sum += in_float4.y * in_float4.y;
      sum += in_float4.z * in_float4.z;
      sum += in_float4.w * in_float4.w;
    }
  }

  const int scalar_start = use_float4 ? pack_off : 0;
  for (int i = scalar_start + tid; i < hidden_size; i += blockDim.x) {
    sum += in_row[i] * in_row[i];
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_sum;
  sum = BlockReduce(temp).Sum(sum);
  if (tid == 0) {
    shared_sum = sum;
  }
  __syncthreads();

  const float scale =
      rsqrtf(shared_sum / static_cast<float>(hidden_size) + eps);

  if (use_float4) {
    const float4 *in_pack = reinterpret_cast<const float4 *>(in_row);
    const float4 *wei_pack = reinterpret_cast<const float4 *>(wei);
    float4 *out_pack = reinterpret_cast<float4 *>(out_row);
    for (int i = tid; i < pack_num; i += blockDim.x) {
      const float4 in_float4 = *(in_pack + i);
      const float4 wei_float4 = *(wei_pack + i);
      *(out_pack + i) = make_float4(scale * in_float4.x * wei_float4.x,
                                    scale * in_float4.y * wei_float4.y,
                                    scale * in_float4.z * wei_float4.z,
                                    scale * in_float4.w * wei_float4.w);
    }
  }

  for (int i = scalar_start + tid; i < hidden_size; i += blockDim.x) {
    out_row[i] = in_row[i] * wei[i] * scale;
  }
}

void LaunchRmsNormBatch(const tensor::Tensor &input,
                        const tensor::Tensor &weight,
                        const tensor::Tensor &output, int32_t batch,
                        int32_t hidden_size, void *stream, float eps) {
  constexpr int threads_num = 128;
  cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
  RmsNormBatchKernel<128><<<batch, threads_num, 0, stream_>>>(
      input.data<float>(), weight.data<float>(),
      const_cast<float *>(output.data<float>()), batch, hidden_size, eps);
  CHECK_CUDA(cudaGetLastError());
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

  constexpr int32_t batch = 1;
  const int32_t hidden_size = static_cast<int32_t>(input.size());
  LaunchRmsNormBatch(input, weight, output, batch, hidden_size, stream, eps);
}

void RmsNormBatchCuda(const tensor::Tensor &input, const tensor::Tensor &weight,
                      const tensor::Tensor &output, void *stream, float eps) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(input.data_type() == base::DataType::kDataTypeFp32);
  CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
  CHECK(output.data_type() == base::DataType::kDataTypeFp32);
  CHECK_EQ(input.dims_size(), 2);
  CHECK_EQ(weight.dims_size(), 1);
  CHECK_EQ(output.dims_size(), 2);
  CHECK_EQ(input.get_dim(0), output.get_dim(0));
  CHECK_EQ(input.get_dim(1), output.get_dim(1));
  CHECK_EQ(input.get_dim(1), weight.get_dim(0));

  const int32_t batch = input.get_dim(0);
  const int32_t hidden_size = input.get_dim(1);
  LaunchRmsNormBatch(input, weight, output, batch, hidden_size, stream, eps);
}
}  // namespace kernel
