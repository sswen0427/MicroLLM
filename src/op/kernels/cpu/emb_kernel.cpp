#include "emb_kernel.h"

#include <cuda_runtime_api.h>

namespace kernel {

void emb_kernel_normal(const tensor::Tensor& input,
                       const tensor::Tensor& weight,
                       const tensor::Tensor& output, int32_t vocab_size,
                       void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  CHECK(weight.device_type() == output.device_type());
  CHECK(input.device_type() == base::DeviceType::kDeviceCPU);

  const auto allocator = base::GetDeviceAllocator(base::DeviceType::kDeviceCPU);
  for (int32_t i = 0; i < input_num; ++i) {
    int32_t token = input.at<int32_t>(i);
    if (token > vocab_size) {
      LOG(FATAL) << "Token index is greater than vocab size.";
    } else {
      float* dest_ptr =
          const_cast<float*>(output.data<float>() + i * weight_dim);
      float* src_ptr =
          const_cast<float*>(weight.data<float>() + (token * weight_dim));
      if (weight.device_type() == base::DeviceType::kDeviceCPU) {
        allocator->memcpy(dest_ptr, src_ptr, weight_dim * sizeof(float),
                          cudaMemcpyHostToHost, nullptr);
      } else {
        LOG(FATAL)
            << "Unknown device type of weight tensor in the embedding layer.";
      }
    }
  }
}

}  // namespace kernel
