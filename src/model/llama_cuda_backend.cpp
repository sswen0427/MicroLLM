#include "model/llama_cuda_backend.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "cuda/emb_kernel.cuh"
#include "cuda/matmul_kernel.cuh"
#include "cuda/rmsnorm_kernel.cuh"
#include "cuda/swiglu_kernel.cuh"
#include "model/llama_backend_util.h"

namespace model {
namespace {

std::vector<int32_t> TensorDims(const tensor::Tensor &tensor) {
  std::vector<int32_t> dims;
  dims.reserve(tensor.dims_size());
  for (int32_t i = 0; i < tensor.dims_size(); ++i) {
    dims.push_back(tensor.get_dim(i));
  }
  return dims;
}

tensor::Tensor CopyVectorToCudaTensor(const std::vector<float> &values) {
  tensor::Tensor tensor = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {static_cast<int32_t>(values.size())},
      base::DeviceType::kDeviceCPU);
  std::copy(values.begin(), values.end(), tensor.data<float>());
  tensor.to_cuda();
  return tensor;
}

std::vector<float> CopyTensorToVector(tensor::Tensor tensor) {
  tensor.to_cpu();
  std::vector<float> values(tensor.size());
  std::copy(tensor.data<float>(), tensor.data<float>() + tensor.size(),
            values.begin());
  return values;
}

} // namespace

base::DeviceType CudaLlamaBackend::device_type() const {
  return base::DeviceType::kDeviceCUDA;
}

void CudaLlamaBackend::Embedding(const tensor::Tensor &weight, int32_t token_id,
                                 std::vector<float> &output) const {
  tensor::Tensor input = tensor::Tensor::allocate(
      base::DataType::kDataTypeInt32, {1}, base::DeviceType::kDeviceCPU);
  input.data<int32_t>()[0] = token_id;
  tensor::Tensor output_tensor = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {weight.get_dim(1)},
      base::DeviceType::kDeviceCUDA);
  kernel::emb_kernel_cu(input, Fp32CudaWeight(weight), output_tensor,
                        weight.get_dim(0), nullptr);
  output = CopyTensorToVector(std::move(output_tensor));
}

void CudaLlamaBackend::RmsNorm(const std::vector<float> &input,
                               const tensor::Tensor &weight, double eps,
                               std::vector<float> &output) const {
  if (std::abs(eps - 1e-5) > 1e-12) {
    cpu_.RmsNorm(input, weight, eps, output);
    return;
  }

  tensor::Tensor input_tensor = CopyVectorToCudaTensor(input);
  tensor::Tensor output_tensor = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {static_cast<int32_t>(input.size())},
      base::DeviceType::kDeviceCUDA);
  kernel::rmsnorm_kernel_cu(input_tensor, Fp32CudaWeight(weight), output_tensor,
                            nullptr);
  output = CopyTensorToVector(std::move(output_tensor));
}

void CudaLlamaBackend::MatVec(const tensor::Tensor &weight,
                              const std::vector<float> &input,
                              std::vector<float> &output) const {
  tensor::Tensor input_tensor = CopyVectorToCudaTensor(input);
  tensor::Tensor output_tensor = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {weight.get_dim(0)},
      base::DeviceType::kDeviceCUDA);
  kernel::matmul_kernel_cu(input_tensor, Fp32CudaWeight(weight), output_tensor,
                           1.0f, nullptr);
  output = CopyTensorToVector(std::move(output_tensor));
}

void CudaLlamaBackend::ApplyRopeToHeads(std::vector<float> &values,
                                        int32_t head_count, int32_t head_size,
                                        int32_t position,
                                        double rope_theta) const {
  cpu_.ApplyRopeToHeads(values, head_count, head_size, position, rope_theta);
}

void CudaLlamaBackend::StoreKvCache(const std::vector<float> &key,
                                    const std::vector<float> &value,
                                    int32_t position, int32_t max_position,
                                    int32_t kv_dim,
                                    std::vector<float> &key_cache,
                                    std::vector<float> &value_cache) const {
  cpu_.StoreKvCache(key, value, position, max_position, kv_dim, key_cache,
                    value_cache);
}

void CudaLlamaBackend::AttentionWithCache(const std::vector<float> &query,
                                          const std::vector<float> &key_cache,
                                          const std::vector<float> &value_cache,
                                          int32_t position, int32_t head_count,
                                          int32_t head_size, int32_t kv_dim,
                                          int32_t kv_mul,
                                          std::vector<float> &output) const {
  cpu_.AttentionWithCache(query, key_cache, value_cache, position, head_count,
                          head_size, kv_dim, kv_mul, output);
}

void CudaLlamaBackend::SwiGlu(const std::vector<float> &gate,
                              const std::vector<float> &up,
                              std::vector<float> &output) const {
  tensor::Tensor gate_tensor = CopyVectorToCudaTensor(gate);
  tensor::Tensor up_tensor = CopyVectorToCudaTensor(up);
  tensor::Tensor output_tensor = tensor::Tensor::allocate(
      base::DataType::kDataTypeFp32, {static_cast<int32_t>(gate.size())},
      base::DeviceType::kDeviceCUDA);
  kernel::swiglu_kernel_cu(gate_tensor, up_tensor, output_tensor, nullptr);
  output = CopyTensorToVector(std::move(output_tensor));
}

void CudaLlamaBackend::AddInPlace(std::vector<float> &left,
                                  const std::vector<float> &right) const {
  cpu_.AddInPlace(left, right);
}

int32_t CudaLlamaBackend::ArgMaxToken(const tensor::Tensor &logits) const {
  return cpu_.ArgMaxToken(logits);
}

const tensor::Tensor &
CudaLlamaBackend::Fp32CudaWeight(const tensor::Tensor &weight) const {
  const auto cached = fp32_cuda_weights_.find(&weight);
  if (cached != fp32_cuda_weights_.end()) {
    return cached->second;
  }

  tensor::Tensor fp32_weight;
  if (weight.data_type() == base::DataType::kDataTypeFp32) {
    fp32_weight = weight.clone();
    if (fp32_weight.device_type() == base::DeviceType::kDeviceCPU) {
      fp32_weight.to_cuda();
    }
  } else {
    fp32_weight = tensor::Tensor::allocate(base::DataType::kDataTypeFp32,
                                           TensorDims(weight),
                                           base::DeviceType::kDeviceCPU);
    for (size_t i = 0; i < weight.size(); ++i) {
      fp32_weight.data<float>()[i] = TensorElementAsFloat(weight, i);
    }
    fp32_weight.to_cuda();
  }

  auto insert_result =
      fp32_cuda_weights_.emplace(&weight, std::move(fp32_weight));
  return insert_result.first->second;
}

} // namespace model
