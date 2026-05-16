#include "rmsnorm_kernel.h"

#include <armadillo>

namespace kernel {
/**
 * @brief CPU implementation of RMSNorm, see
 * https://zhuanlan.zhihu.com/p/685181632
 *
 * Computes:
 *   output = weight * input / sqrt(mean(input^2) + eps)
 *
 * `eps` is 1e-6 for QWEN2/QWEN3 and 1e-5 otherwise.
 * All tensors must be non-empty, on CPU, and stored as float.
 */
void rmsnorm_kernel_cpu(const tensor::Tensor& input,
                        const tensor::Tensor& weight,
                        const tensor::Tensor& output,
                        [[maybe_unused]] void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCPU &&
        weight.device_type() == base::DeviceType::kDeviceCPU &&
        output.device_type() == base::DeviceType::kDeviceCPU);

  const float* in_ptr = input.ptr<float>();
  const float* wei_ptr = weight.ptr<float>();
  const float* out_ptr = output.ptr<float>();
  const int32_t dim = static_cast<int32_t>(input.size());

  arma::fvec in_tensor(const_cast<float*>(in_ptr), dim, false, true);
  arma::fvec out_tensor(const_cast<float*>(out_ptr), dim, false, true);
  arma::fvec wei_tensor(const_cast<float*>(wei_ptr), dim, false, true);

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  const float eps = 1e-6f;
#else
  const float eps = 1e-5f;
#endif

  const float mean = arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + eps;
  const float rsqrt = 1.f / std::sqrt(mean);
  out_tensor = wei_tensor % (rsqrt * in_tensor);
}
}  // namespace kernel