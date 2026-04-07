#ifndef MICROLLM_INCLUDE_BASE_CUDA_CONFIG_H
#define MICROLLM_INCLUDE_BASE_CUDA_CONFIG_H
#include <cuda_runtime_api.h>
#include <driver_types.h>

namespace base {
struct CudaConfig {
  cudaStream_t stream;

  ~CudaConfig() {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
};
}  // namespace base

#endif  // MICROLLM_INCLUDE_BASE_CUDA_CONFIG_H
