#include <cuda_runtime_api.h>

#include <cmath>

#include "cuda/cuda_check.h"
#include "cuda/llama.cuh"

namespace kernel {
namespace {

/**
 * @brief Applies RoPE rotation to query/key vectors in place.
 *
 * Tensor shapes in logical row-major layout:
 *   values: [seq_len, head_count * head_size]
 *
 * Each thread rotates one pair in one attention head:
 *   [x0, x1] -> [x0 * cos(theta) - x1 * sin(theta),
 *                x0 * sin(theta) + x1 * cos(theta)]
 * where theta = (start_position + token_idx) *
 *               rope_theta^(-2i / head_size).
 *
 * Single-token decode uses this same kernel with seq_len = 1.
 */
__global__ void RopeInPlaceBatchKernel(float *values, int32_t seq_len,
                                       int32_t head_count, int32_t head_size,
                                       int32_t start_position,
                                       float rope_theta) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t half_head_size = head_size / 2;
  const int32_t pairs_per_token = head_count * half_head_size;
  const int32_t total_pairs = seq_len * pairs_per_token;
  if (idx >= total_pairs) {
    return;
  }

  const int32_t token_idx = idx / pairs_per_token;
  const int32_t pair_idx = idx % pairs_per_token;
  const int32_t head = pair_idx / half_head_size;
  const int32_t i = pair_idx % half_head_size;
  const int32_t token_offset = token_idx * head_count * head_size;
  const int32_t head_offset = token_offset + head * head_size;
  const int32_t first = head_offset + i;
  const int32_t second = first + half_head_size;

  const float freq = powf(
      rope_theta, -static_cast<float>(2 * i) / static_cast<float>(head_size));
  const float angle = static_cast<float>(start_position + token_idx) * freq;
  float sin_value;
  float cos_value;
  sincosf(angle, &sin_value, &cos_value);

  const float x0 = values[first];
  const float x1 = values[second];
  values[first] = x0 * cos_value - x1 * sin_value;
  values[second] = x0 * sin_value + x1 * cos_value;
}

/**
 * @brief Stores key/value vectors into the KV cache.
 *
 * Tensor shapes in logical row-major layout:
 *   key:         [seq_len, kv_dim]
 *   value:       [seq_len, kv_dim]
 *   key_cache:   [max_seq_len, kv_dim]
 *   value_cache: [max_seq_len, kv_dim]
 *
 * Each thread copies one element:
 *   key_cache[start_position + token_idx, col] = key[token_idx, col]
 *   value_cache[start_position + token_idx, col] = value[token_idx, col]
 *
 * Single-token decode uses this same kernel with seq_len = 1.
 */
__global__ void StoreKvCacheBatchKernel(const float *key, const float *value,
                                        float *key_cache, float *value_cache,
                                        int32_t start_position, int32_t seq_len,
                                        int32_t kv_dim) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t total = seq_len * kv_dim;
  if (idx >= total) {
    return;
  }
  const int32_t token_idx = idx / kv_dim;
  const int32_t col = idx % kv_dim;
  const int32_t cache_offset = (start_position + token_idx) * kv_dim + col;
  key_cache[cache_offset] = key[idx];
  value_cache[cache_offset] = value[idx];
}

/**
 * @brief Computes the scaled dot-product score for one query head and token.
 *
 * Tensor shapes in logical row-major layout:
 *   query:     [head_count, head_size]
 *   key_cache: [max_seq_len, kv_dim], kv_dim = kv_head_count * head_size
 *
 * kv_mul maps multiple query heads to one KV head for GQA/MQA:
 *   kv_head = query_head / kv_mul
 *   score = dot(query[query_head], key_cache[token, kv_head]) / sqrt(head_size)
 *
 * @param query Current token query vectors.
 * @param key_cache Cached key vectors for previous and current tokens.
 * @param token Cache token index to score against.
 * @param head Query head index.
 * @param head_size Number of values in one attention head.
 * @param kv_dim Number of key values stored per token.
 * @param kv_mul Number of query heads sharing one KV head.
 */
__device__ float AttentionScore(const float *query, const float *key_cache,
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

/**
 * @brief Computes causal attention using cached keys and values.
 *
 * Tensor shapes in logical row-major layout:
 *   query:       [seq_len, head_count * head_size]
 *   key_cache:   [max_seq_len, kv_dim], kv_dim = kv_head_count * head_size
 *   value_cache: [max_seq_len, kv_dim], kv_dim = kv_head_count * head_size
 *   output:      [seq_len, head_count * head_size]
 *
 * Each block handles one (query token, query head), and each thread writes one
 * output dimension in that head. Query token `token_idx` attends over cached
 * positions [0, start_position + token_idx]:
 *   score_t = dot(query[token_idx, head], key_cache[t, kv_head]) /
 *             sqrt(head_size)
 *   prob_t = softmax(score_t)
 *   output[token_idx, head, dim] =
 *       sum_t prob_t * value_cache[t, kv_head, dim]
 *
 * Single-token decode uses this same kernel with seq_len = 1.
 */
__global__ void AttentionWithCacheBatchKernel(
    const float *query, const float *key_cache, const float *value_cache,
    float *output, int32_t start_position, int32_t seq_len, int32_t head_count,
    int32_t head_size, int32_t kv_dim, int32_t kv_mul) {
  const int32_t head = blockIdx.x;
  const int32_t token_idx = blockIdx.y;
  const int32_t dim = threadIdx.x;
  if (head >= head_count || token_idx >= seq_len || dim >= head_size) {
    return;
  }

  const int32_t position = start_position + token_idx;
  const float *query_token = query + token_idx * head_count * head_size;

  float max_score = -INFINITY;
  for (int32_t token = 0; token <= position; ++token) {
    const float score = AttentionScore(query_token, key_cache, token, head,
                                       head_size, kv_dim, kv_mul);
    max_score = fmaxf(max_score, score);
  }

  float denom = 0.0f;
  for (int32_t token = 0; token <= position; ++token) {
    const float score = AttentionScore(query_token, key_cache, token, head,
                                       head_size, kv_dim, kv_mul);
    denom += expf(score - max_score);
  }

  const int32_t kv_head = head / kv_mul;
  float sum = 0.0f;
  for (int32_t token = 0; token <= position; ++token) {
    const float score = AttentionScore(query_token, key_cache, token, head,
                                       head_size, kv_dim, kv_mul);
    const float prob = expf(score - max_score) / denom;
    const int32_t cache_offset = token * kv_dim + kv_head * head_size + dim;
    sum += prob * value_cache[cache_offset];
  }

  const int32_t output_offset =
      token_idx * head_count * head_size + head * head_size + dim;
  output[output_offset] = sum;
}

cudaStream_t AsCudaStream(void *stream) {
  return static_cast<cudaStream_t>(stream);
}

void LaunchRopeInPlace(tensor::Tensor &values, int32_t seq_len,
                       int32_t head_count, int32_t head_size,
                       int32_t start_position, double rope_theta,
                       void *stream) {
  constexpr int32_t threads = 128;
  const int32_t total_pairs = seq_len * head_count * (head_size / 2);
  const int32_t blocks = (total_pairs + threads - 1) / threads;
  RopeInPlaceBatchKernel<<<blocks, threads, 0, AsCudaStream(stream)>>>(
      values.data<float>(), seq_len, head_count, head_size, start_position,
      static_cast<float>(rope_theta));
  CHECK_CUDA(cudaGetLastError());
}

void LaunchStoreKvCache(const tensor::Tensor &key, const tensor::Tensor &value,
                        tensor::Tensor &key_cache,
                        tensor::Tensor &value_cache, int32_t start_position,
                        int32_t seq_len, int32_t kv_dim, void *stream) {
  constexpr int32_t threads = 128;
  const int32_t total = seq_len * kv_dim;
  const int32_t blocks = (total + threads - 1) / threads;
  StoreKvCacheBatchKernel<<<blocks, threads, 0, AsCudaStream(stream)>>>(
      key.data<float>(), value.data<float>(), key_cache.data<float>(),
      value_cache.data<float>(), start_position, seq_len, kv_dim);
  CHECK_CUDA(cudaGetLastError());
}

void LaunchAttentionWithCache(const tensor::Tensor &query,
                              const tensor::Tensor &key_cache,
                              const tensor::Tensor &value_cache,
                              const tensor::Tensor &output,
                              int32_t start_position, int32_t seq_len,
                              int32_t head_count, int32_t head_size,
                              int32_t kv_dim, int32_t kv_mul, void *stream) {
  const dim3 grid(head_count, seq_len);
  AttentionWithCacheBatchKernel<<<grid, head_size, 0, AsCudaStream(stream)>>>(
      query.data<float>(), key_cache.data<float>(), value_cache.data<float>(),
      const_cast<float *>(output.data<float>()), start_position, seq_len,
      head_count, head_size, kv_dim, kv_mul);
  CHECK_CUDA(cudaGetLastError());
}

}  // namespace

void RopeInPlaceCuda(tensor::Tensor &values, int32_t head_count,
                     int32_t head_size, int32_t start_position,
                     double rope_theta, void *stream) {
  CHECK(!values.is_empty());
  CHECK(values.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(values.data_type() == base::DataType::kDataTypeFp32);
  CHECK_EQ(values.dims_size(), 2);
  CHECK_EQ(values.get_dim(1), head_count * head_size);
  CHECK_EQ(head_size % 2, 0);
  CHECK_GE(start_position, 0);

  const int32_t seq_len = values.get_dim(0);
  LaunchRopeInPlace(values, seq_len, head_count, head_size, start_position,
                    rope_theta, stream);
}

void StoreKvCacheCuda(const tensor::Tensor &key, const tensor::Tensor &value,
                      tensor::Tensor &key_cache, tensor::Tensor &value_cache,
                      int32_t start_position, int32_t kv_dim, void *stream) {
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
  CHECK(key_cache.data_type() == base::DataType::kDataTypeFp32);
  CHECK(value_cache.data_type() == base::DataType::kDataTypeFp32);
  CHECK_EQ(key.dims_size(), 2);
  CHECK_EQ(value.dims_size(), 2);
  CHECK_EQ(key_cache.dims_size(), 2);
  CHECK_EQ(value_cache.dims_size(), 2);
  CHECK_EQ(key.get_dim(0), value.get_dim(0));
  CHECK_EQ(key.get_dim(1), kv_dim);
  CHECK_EQ(value.get_dim(1), kv_dim);
  CHECK_EQ(key_cache.get_dim(1), kv_dim);
  CHECK_EQ(value_cache.get_dim(1), kv_dim);
  CHECK_GE(start_position, 0);
  CHECK_LE(start_position + key.get_dim(0), key_cache.get_dim(0));
  CHECK_EQ(key_cache.get_dim(0), value_cache.get_dim(0));

  const int32_t seq_len = key.get_dim(0);
  LaunchStoreKvCache(key, value, key_cache, value_cache, start_position,
                     seq_len, kv_dim, stream);
}

void AttentionWithCacheCuda(const tensor::Tensor &query,
                            const tensor::Tensor &key_cache,
                            const tensor::Tensor &value_cache,
                            const tensor::Tensor &output,
                            int32_t start_position,
                            int32_t head_count, int32_t head_size,
                            int32_t kv_dim, int32_t kv_mul, void *stream) {
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
  CHECK_EQ(query.dims_size(), 2);
  CHECK_EQ(output.dims_size(), 2);
  CHECK_EQ(key_cache.dims_size(), 2);
  CHECK_EQ(value_cache.dims_size(), 2);
  CHECK_EQ(query.get_dim(0), output.get_dim(0));
  CHECK_EQ(query.get_dim(1), head_count * head_size);
  CHECK_EQ(output.get_dim(1), head_count * head_size);
  CHECK_EQ(key_cache.get_dim(1), kv_dim);
  CHECK_EQ(value_cache.get_dim(1), kv_dim);
  CHECK_EQ(key_cache.get_dim(0), value_cache.get_dim(0));
  CHECK_GE(start_position, 0);
  CHECK_LT(start_position + query.get_dim(0), key_cache.get_dim(0) + 1);
  CHECK_GT(kv_mul, 0);
  CHECK_LE(head_size, 1024);

  LaunchAttentionWithCache(query, key_cache, value_cache, output,
                           start_position, query.get_dim(0), head_count,
                           head_size, kv_dim, kv_mul, stream);
}

}  // namespace kernel
