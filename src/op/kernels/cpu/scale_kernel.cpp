#include "scale_kernel.h"

#include <armadillo>

namespace kernel {
void scale_inplace_cpu(float scale, const tensor::Tensor& tensor,
                       [[maybe_unused]] void* stream) {
  CHECK(tensor.is_empty() == false);
  arma::fvec tensor_mat(const_cast<float*>(tensor.data<float>()), tensor.size(),
                        false, true);
  tensor_mat *= scale;
}
}  // namespace kernel