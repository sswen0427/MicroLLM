#include "model/llama_backend.h"

#include <glog/logging.h>

#include <memory>

#include "model/llama_cpu_backend.h"
#include "model/llama_cuda_backend.h"

namespace model {

std::unique_ptr<LlamaBackend> CreateLlamaBackend(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return std::make_unique<CpuLlamaBackend>();
  }
  if (device_type == base::DeviceType::kDeviceCUDA) {
    return std::make_unique<CudaLlamaBackend>();
  }
  LOG(FATAL) << "Unsupported LLaMA backend device type: "
             << static_cast<int>(device_type);
  return nullptr;
}

}  // namespace model
