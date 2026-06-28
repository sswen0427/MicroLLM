#include <cuda_runtime_api.h>

#include <cmath>

#include "cuda/cuda_check.h"
#include "cuda/llama_kernel.cuh"

namespace kernel {
namespace {

__global__ void rope_inplace_kernel_fp32(float *values, int32_t head_count,
                                         int32_t head_size, int32_t position,
                                         float rope_theta) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t half_head_size = head_size / 2;
  const int32_t total_pairs = head_count * half_head_size;
  if (idx >= total_pairs) {
    return;
  }

  const int32_t head = idx / half_head_size;
  const int32_t i = idx % half_head_size;
  const int32_t head_offset = head * head_size;
  const int32_t first = head_offset + i;
  const int32_t second = first + half_head_size;

  const float freq = powf(
      rope_theta, -static_cast<float>(2 * i) / static_cast<float>(head_size));
  const float angle = static_cast<float>(position) * freq;
  float sin_value;
  float cos_value;
  sincosf(angle, &sin_value, &cos_value);

  const float x0 = values[first];
  const float x1 = values[second];
  values[first] = x0 * cos_value - x1 * sin_value;
  values[second] = x0 * sin_value + x1 * cos_value;
}

__global__ void store_kv_cache_kernel_fp32(const float *key, const float *value,
                                           float *key_cache, float *value_cache,
                                           int32_t position, int32_t kv_dim) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= kv_dim) {
    return;
  }
  const int32_t cache_offset = position * kv_dim + idx;
  key_cache[cache_offset] = key[idx];
  value_cache[cache_offset] = value[idx];
}

__device__ float attention_score(const float *query, const float *key_cache,
                                 int32_t token, int32_t head, int32_t head_size,
                                 int32_t kv_dim, int32_t kv_mul) {
  const int32_t kv_head = head / kv_mul;
  const int32_t query_offset = head * head_size;
  const int32_t cache_offset = token * kv_dim + kv_head * head_size;

  float score = 0.0f;
  for (int32_t i = 0; i < head_size; ++i) {
    score += query[query_offset + i] * key_cache[cache_offset + i];
  }
  return score * rsqrtf(static_cast<float>(head_size));
}

__global__ void attention_with_cache_kernel_fp32(
    const float *query, const float *key_cache, const float *value_cache,
    float *output, int32_t position, int32_t head_count, int32_t head_size,
    int32_t kv_dim, int32_t kv_mul) {
  const int32_t head = blockIdx.x;
  const int32_t dim = threadIdx.x;
  if (head >= head_count || dim >= head_size) {
    return;
  }

  float max_score = -INFINITY;
  for (int32_t token = 0; token <= position; ++token) {
    const float score = attention_score(query, key_cache, token, head,
                                        head_size, kv_dim, kv_mul);
    max_score = fmaxf(max_score, score);
  }

  float denom = 0.0f;
  for (int32_t token = 0; token <= position; ++token) {
    const float score = attention_score(query, key_cache, token, head,
                                        head_size, kv_dim, kv_mul);
    denom += expf(score - max_score);
  }

  const int32_t kv_head = head / kv_mul;
  float sum = 0.0f;
  for (int32_t token = 0; token <= position; ++token) {
    const float score = attention_score(query, key_cache, token, head,
                                        head_size, kv_dim, kv_mul);
    const float prob = expf(score - max_score) / denom;
    const int32_t cache_offset = token * kv_dim + kv_head * head_size + dim;
    sum += prob * value_cache[cache_offset];
  }

  output[head * head_size + dim] = sum;
}

cudaStream_t AsCudaStream(void *stream) {
  return static_cast<cudaStream_t>(stream);
}

}  // namespace

void rope_inplace_kernel_cu(tensor::Tensor &values, int32_t head_count,
                            int32_t head_size, int32_t position,
                            double rope_theta, void *stream) {
  CHECK(!values.is_empty());
  CHECK(values.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(values.data_type() == base::DataType::kDataTypeFp32);
  CHECK_EQ(static_cast<int32_t>(values.size()), head_count * head_size);
  CHECK_EQ(head_size % 2, 0);

  constexpr int32_t threads = 128;
  const int32_t total_pairs = head_count * (head_size / 2);
  const int32_t blocks = (total_pairs + threads - 1) / threads;
  rope_inplace_kernel_fp32<<<blocks, threads, 0, AsCudaStream(stream)>>>(
      values.data<float>(), head_count, head_size, position,
      static_cast<float>(rope_theta));
  CHECK_CUDA(cudaGetLastError());
}

void store_kv_cache_kernel_cu(const tensor::Tensor &key,
                              const tensor::Tensor &value,
                              tensor::Tensor &key_cache,
                              tensor::Tensor &value_cache, int32_t position,
                              int32_t kv_dim, void *stream) {
  CHECK(!key.is_empty());
  CHECK(!value.is_empty());
  CHECK(!key_cache.is_empty());
  CHECK(!value_cache.is_empty());
  CHECK(key.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(value.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(key_cache.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(value_cache.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(key.data_type() == base::DataType::kDataTypeFp32);
  CHECK(value.data_type() == base::DataType::kDataTypeFp32);
  CHECK_EQ(static_cast<int32_t>(key.size()), kv_dim);
  CHECK_EQ(static_cast<int32_t>(value.size()), kv_dim);

  constexpr int32_t threads = 128;
  const int32_t blocks = (kv_dim + threads - 1) / threads;
  store_kv_cache_kernel_fp32<<<blocks, threads, 0, AsCudaStream(stream)>>>(
      key.data<float>(), value.data<float>(), key_cache.data<float>(),
      value_cache.data<float>(), position, kv_dim);
  CHECK_CUDA(cudaGetLastError());
}

void attention_with_cache_kernel_cu(const tensor::Tensor &query,
                                    const tensor::Tensor &key_cache,
                                    const tensor::Tensor &value_cache,
                                    const tensor::Tensor &output,
                                    int32_t position, int32_t head_count,
                                    int32_t head_size, int32_t kv_dim,
                                    int32_t kv_mul, void *stream) {
  CHECK(!query.is_empty());
  CHECK(!key_cache.is_empty());
  CHECK(!value_cache.is_empty());
  CHECK(!output.is_empty());
  CHECK(query.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(key_cache.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(value_cache.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(query.data_type() == base::DataType::kDataTypeFp32);
  CHECK(key_cache.data_type() == base::DataType::kDataTypeFp32);
  CHECK(value_cache.data_type() == base::DataType::kDataTypeFp32);
  CHECK(output.data_type() == base::DataType::kDataTypeFp32);
  CHECK_EQ(static_cast<int32_t>(query.size()), head_count * head_size);
  CHECK_EQ(static_cast<int32_t>(output.size()), head_count * head_size);
  CHECK_GT(kv_mul, 0);

  attention_with_cache_kernel_fp32<<<head_count, head_size, 0,
                                     AsCudaStream(stream)>>>(
      query.data<float>(), key_cache.data<float>(), value_cache.data<float>(),
      const_cast<float *>(output.data<float>()), position, head_count,
      head_size, kv_dim, kv_mul);
  CHECK_CUDA(cudaGetLastError());
}

}  // namespace kernel
