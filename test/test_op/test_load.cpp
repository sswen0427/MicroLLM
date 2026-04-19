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
  auto* weight_data = reinterpret_cast<float*>(config + 1);

  for (int i = 0; i < config->dim * config->hidden_dim; ++i) {
    EXPECT_EQ(*(weight_data + i), float(i));
  }
  /**                                  1
   *    1 2 3 4 5 6 ... 1024           1
   *                                   1
   *                                   1
   */
  auto wq = std::make_shared<op::MatmulLayer>(
      base::DeviceType::kDeviceCPU, config->dim, config->hidden_dim, false);
  float* in = new float[config->hidden_dim];
  for (int i = 0; i < config->hidden_dim; ++i) {
    in[i] = 1.f;
  }

  float* out = new float[config->dim];
  for (int i = 0; i < config->dim; ++i) {
    out[i] = 0.f;
  }
  tensor::Tensor tensor = tensor::Tensor::from_external(
      base::DataType::kDataTypeFp32, {config->hidden_dim}, in);
  tensor.set_device_type(base::DeviceType::kDeviceCPU);

  tensor::Tensor out_tensor = tensor::Tensor::from_external(
      base::DataType::kDataTypeFp32, {config->dim}, out);
  out_tensor.set_device_type(base::DeviceType::kDeviceCPU);

  wq->set_input(0, tensor);
  wq->set_output(0, out_tensor);
  wq->set_weight(0, {config->dim, config->hidden_dim}, weight_data,
                 base::DeviceType::kDeviceCPU);
  wq->forward();  // 完成一个计算

  /** python code:
   *  w = np.arange(0,128 * 16).reshape(16, 128)
   *  input = np.ones(128)
   *  out = w@input
   */
  ASSERT_EQ(out[0], 8128);
  ASSERT_EQ(out[1], 24512);
  ASSERT_EQ(out[14], 237504);
  ASSERT_EQ(out[15], 253888);

  delete[] in;
  delete[] out;
}