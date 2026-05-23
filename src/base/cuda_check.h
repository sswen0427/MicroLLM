#pragma once

#include <cuda_runtime_api.h>
#include <glog/logging.h>

namespace base {

inline void CheckCuda(cudaError_t error, const char* expression) {
  CHECK_EQ(error, cudaSuccess)
      << expression << " failed: " << cudaGetErrorString(error);
}

}  // namespace base

#define CHECK_CUDA(expression)                        \
  do {                                                \
    ::base::CheckCuda((expression), #expression);     \
  } while (false)
