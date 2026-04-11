#include <cuda_runtime_api.h>
#include <glog/logging.h>

#include "alloc.h"

namespace base {
CUDADeviceAllocator::CUDADeviceAllocator()
    : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
  int id = -1;
  cudaError_t state = cudaGetDevice(&id);
  CHECK(state == cudaSuccess) << "cudaGetDevice failed!";
  // use big_buffers_map_ if byte_size > 1MB, else use small_buffers_map_
  if (byte_size > 1024 * 1024) {
    // Step1.1: Try to allocate from big buffers
    auto& big_buffers = big_buffers_map_[id];
    int sel_id = -1;
    for (int i = 0; i < big_buffers.size(); i++) {
      if (big_buffers[i].size >= byte_size && !big_buffers[i].busy &&
          big_buffers[i].size - byte_size < 1 * 1024 * 1024) {
        if (sel_id == -1 || big_buffers[sel_id].size > big_buffers[i].size) {
          sel_id = i;
        }
      }
    }
    if (sel_id != -1) {
      big_buffers[sel_id].busy = true;
      return big_buffers[sel_id].data;
    }

    // Step1.2: If no big buffer is available, allocate from device memory
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    if (state != cudaSuccess) {
      LOG(ERROR) << "cudaMalloc failed! error code: "
                 << cudaGetErrorString(state);
      return nullptr;
    }
    big_buffers.emplace_back(ptr, byte_size, true);
    return ptr;
  } else {
    auto& small_buffers = small_buffers_map_[id];
    for (int i = 0; i < small_buffers.size(); i++) {
      if (small_buffers[i].size >= byte_size && !small_buffers[i].busy) {
        small_buffers[i].busy = true;
        small_buffers_idle_bytes_[id] -= small_buffers[i].size;
        return small_buffers[i].data;
      }
    }
    // Step2.2: If no cuda buffer is available, allocate from device memory
    void* ptr = nullptr;
    if (cudaSuccess != cudaMalloc(&ptr, byte_size)) {
      LOG(ERROR) << "cudaMalloc failed! error code: "
                 << cudaGetErrorString(state);
      return nullptr;
    }
    small_buffers.emplace_back(ptr, byte_size, true);
    return ptr;
  }
}

void CUDADeviceAllocator::release(void* ptr) const {
  CHECK(ptr) << "CUDADeviceAllocator::release(): ptr is nullptr";
  cudaError_t state = cudaSuccess;
  int current_id = -1;
  cudaGetDevice(&current_id);
  for (auto& [id, small_buffers] : small_buffers_map_) {
    if (small_buffers_idle_bytes_[id] > 1024 * 1024 * 1024) {
      std::vector<CUDAMemoryBuffer> temp;
      for (int i = 0; i < small_buffers.size(); i++) {
        if (!small_buffers[i].busy) {
          state = cudaSetDevice(id);
          state = cudaFree(small_buffers[i].data);
          CHECK(state == cudaSuccess)
              << "Error: CUDA error when release memory on device " << id;
        } else {
          temp.push_back(small_buffers[i]);
        }
      }
      small_buffers.clear();
      small_buffers = temp;
      small_buffers_idle_bytes_[id] = 0;
    }
  }
  cudaGetDevice(&current_id);

  for (auto& [id, small_buffers] : small_buffers_map_) {
    for (int i = 0; i < small_buffers.size(); i++) {
      if (small_buffers[i].data == ptr) {
        small_buffers_idle_bytes_[id] += small_buffers[i].size;
        small_buffers[i].busy = false;
        return;
      }
    }
  }
  for (auto& [id, big_buffers] : big_buffers_map_) {
    for (int i = 0; i < big_buffers.size(); i++) {
      if (big_buffers[i].data == ptr) {
        big_buffers[i].busy = false;
        return;
      }
    }
  }
  state = cudaFree(ptr);
  CHECK(state == cudaSuccess)
      << "Error: CUDA error when release memory on device";
}
std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance =
    nullptr;

}  // namespace base
