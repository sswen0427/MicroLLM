#include "mha_kernel.h"

#include <cmath>
#include <vector>

#include "op/kernels/kernels_interface.h"

namespace kernel {
namespace {

tensor::Tensor MakeExternalTensor(base::DataType data_type,
                                  const std::vector<int32_t>& dims, void* ptr,
                                  base::DeviceType device_type) {
  CHECK_NE(device_type, base::DeviceType::kDeviceUnknown);
  if (device_type == base::DeviceType::kDeviceCUDA) {
    return tensor::Tensor::from_external_cuda(data_type, dims, ptr);
  }
  return tensor::Tensor::from_external_cpu(data_type, dims, ptr);
}

}  // namespace

void mha_kernel(int32_t pos, int32_t head_num, int32_t layer_index,
                int32_t seq_len, int32_t kv_dim, int32_t kv_mul,
                int32_t head_size, const tensor::Tensor& mha_out,
                const tensor::Tensor& query_tensor,
                const tensor::Tensor& score_tensor,
                const tensor::Tensor& key_cache_tensor,
                const tensor::Tensor& value_cache_tensor,
                base::DeviceType device_type, base::CudaConfig* config) {
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float scale = 1.f / std::sqrt(static_cast<float>(head_size));

  std::shared_ptr<base::DeviceAllocator> allocator;
  if (device_type == base::DeviceType::kDeviceCPU) {
    allocator = base::GetDeviceAllocator(base::DeviceType::kDeviceCPU);
  } else {
    allocator = base::GetDeviceAllocator(base::DeviceType::kDeviceCUDA);
  }
  for (int32_t h = 0; h < head_num; ++h) {
    float* score_head_addr =
        const_cast<float*>(score_tensor.data<float>() + h * seq_len);
    float* query_head_addr =
        const_cast<float*>(query_tensor.data<float>() + h * head_size);

    tensor::Tensor query_mat = MakeExternalTensor(
        base::DataType::kDataTypeFp32, {head_size}, query_head_addr,
        device_type);

    for (int32_t t = 0; t <= pos; t++) {
      int32_t cache_offset = t * kv_dim + (h / kv_mul) * head_size;
      const float* key_head_addr =
          key_cache_tensor.data<float>() + layer_offset + cache_offset;
      tensor::Tensor key_mat = MakeExternalTensor(
          base::DataType::kDataTypeFp32, {1, head_size},
          const_cast<float*>(key_head_addr), device_type);

      tensor::Tensor score_mat = MakeExternalTensor(
          base::DataType::kDataTypeFp32, {1}, score_head_addr + t,
          device_type);
      get_matmul_kernel(device_type)(query_mat, key_mat, score_mat, scale,
                                     config);
    }

    tensor::Tensor score_head_tensor = MakeExternalTensor(
        base::DataType::kDataTypeFp32, {pos + 1}, score_head_addr,
        device_type);
    get_softmax_kernel(device_type)(score_head_tensor,
                                    config ? config->stream : nullptr);

    float* output_head_ptr =
        const_cast<float*>(mha_out.data<float>()) + h * head_size;
    allocator->memset_zero(output_head_ptr, sizeof(float) * head_size,
                           config ? config->stream : nullptr);
    tensor::Tensor output_tensor = MakeExternalTensor(
        base::DataType::kDataTypeFp32, {head_size}, output_head_ptr,
        device_type);

    int32_t cache_offset = (h / kv_mul) * head_size;
    float* value_head_addr =
        const_cast<float*>(value_cache_tensor.data<float>()) + layer_offset +
        cache_offset;
    tensor::Tensor value_tensor = MakeExternalTensor(
        base::DataType::kDataTypeFp32, {head_size}, value_head_addr,
        device_type);
    get_scale_sum_kernel(device_type)(value_tensor, score_head_tensor,
                                      output_tensor, pos, head_size, kv_dim,
                                      config ? config->stream : nullptr);
  }
}
}  // namespace kernel
