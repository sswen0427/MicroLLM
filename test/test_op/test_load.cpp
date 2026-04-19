#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <model/config.h>
#include <sys/mman.h>

#include "base/alias.h"
#include "base/buffer.h"
#include "op/matmul.h"
#include "tensor/tensor.h"

TEST(LoadTest, Matmul) {
  Path root_path = ROOT_PATH;
  Path bin_path = root_path / "data/test.bin";

  int32_t fd = open(bin_path.c_str(), O_RDONLY);
  EXPECT_NE(fd, -1);

  struct stat st;
  fstat(fd, &st);
  std::size_t file_size = st.st_size;
  EXPECT_EQ(file_size, 8220);

  void* data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  EXPECT_NE(data, MAP_FAILED);

  auto* config = static_cast<model::ModelConfig*>(data);
  auto* raw_weight_data = reinterpret_cast<float*>(config + 1);
  /**
   *
   * [CRITICAL HARDWARE ALIGNMENT FIX]
   * We cannot use the raw pointer from `mmap` directly here.
   * The raw weight data immediately follows the `ModelConfig` struct (28
   * bytes). This 28-byte offset breaks the strict 64-byte memory alignment
   * requirement of the underlying BLAS library (e.g., OpenBLAS with AVX-512
   * instructions). Passing the misaligned raw pointer to the matmul kernel will
   * trigger a hardware exception (General Protection Fault) and result in a
   * SIGSEGV.
   *
   * Therefore, we use std::vector to allocate a fresh, guaranteed-aligned
   * memory block and copy the weights into it before passing to the tensor.
   */
  std::vector<float> aligned_weight(config->dim * config->hidden_dim);
  std::memcpy(aligned_weight.data(), raw_weight_data,
              aligned_weight.size() * sizeof(float));

  for (int i = 0; i < aligned_weight.size(); ++i) {
    EXPECT_EQ(aligned_weight[i], float(i));
  }
  /**                                  1
   *    1 2 3 4 5 6 ... 2048           1
   *                                   1
   *                                   1
   */
  auto wq = std::make_shared<op::MatmulLayer>(
      base::DeviceType::kDeviceCPU, config->dim, config->hidden_dim, false);
  std::vector<float> in(config->hidden_dim, 1.f);
  std::vector<float> out(config->hidden_dim, 0.f);

  tensor::Tensor tensor = tensor::Tensor::from_external(
      base::DataType::kDataTypeFp32, {config->hidden_dim}, in.data());
  tensor.set_device_type(base::DeviceType::kDeviceCPU);

  tensor::Tensor out_tensor = tensor::Tensor::from_external(
      base::DataType::kDataTypeFp32, {config->dim}, out.data());
  out_tensor.set_device_type(base::DeviceType::kDeviceCPU);

  wq->set_input(0, tensor);
  wq->set_output(0, out_tensor);
  wq->set_weight(0, {config->dim, config->hidden_dim}, aligned_weight.data(),
                 base::DeviceType::kDeviceCPU);
  wq->forward();  // 完成一个计算

  /** python code:
   *  w = np.arange(0,128 * 16).reshape(16, 128)
   *  input = np.ones(128)
   *  out = w@input
   */
  EXPECT_EQ(out[0], 8128);
  EXPECT_EQ(out[1], 24512);
  EXPECT_EQ(out[14], 237504);
  EXPECT_EQ(out[15], 253888);
}