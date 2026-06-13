#pragma once

#include <cuda_runtime_api.h>
#include <glog/logging.h>

namespace cuda_util {

inline void CheckCuda(cudaError_t error, const char *expression) {
  CHECK_EQ(error, cudaSuccess)
      << expression << " failed: " << cudaGetErrorString(error);
}

}  // namespace cuda_util

#define CHECK_CUDA(expression)                         \
  do {                                                 \
    ::cuda_util::CheckCuda((expression), #expression); \
  } while (false)
