#ifndef MICROLLM_INCLUDE_OP_LAYER_H
#define MICROLLM_INCLUDE_OP_LAYER_H
#include <string>

#include "base/base.h"
#include "tensor/tensor.h"

namespace op {

enum class LayerType : uint8_t {
  kLayerUnknown = 0,
  kLayerLinear = 1,
  kLayerEncode = 2,
  kLayerEmbedding = 3,
  kLayerRMSNorm = 4,
  kLayerMatmul = 5,
  kLayerRoPe = 6,
  kLayerMHA = 7,
  kLayerSoftmax = 8,
  kLayerAdd = 9,
  kLayerSwiGLU = 10,
};

class BaseLayer {
 public:
  explicit BaseLayer(base::DeviceType device_type, LayerType layer_type,
                     base::DataType data_type, std::string layer_name = "");

  virtual ~BaseLayer() = default;

  virtual base::Status forward() = 0;

  virtual base::Status forward(const std::vector<tensor::Tensor>& inputs,
                               std::vector<tensor::Tensor>& outputs) = 0;

  virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

  virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

  virtual size_t input_size() const = 0;

  virtual size_t output_size() const = 0;

  virtual base::Status check() const = 0;

  virtual tensor::Tensor& get_input(int32_t idx) = 0;

  virtual tensor::Tensor& get_output(int32_t idx) = 0;

  virtual const tensor::Tensor& get_input(int32_t idx) const = 0;

  virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

  virtual base::Status set_weight(int32_t idx,
                                  const tensor::Tensor& weight) = 0;

  virtual base::Status set_weight(
      int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
      base::DeviceType device_type = base::DeviceType::kDeviceUnknown) = 0;

  base::DataType data_type() const;

  LayerType layer_type() const;

  const std::string& get_layer_name() const;

  void set_layer_name(const std::string& layer_name);

  base::DeviceType device_type() const;

  void set_device_type(base::DeviceType device_type);

 protected:
  std::string layer_name_;
  LayerType layer_type_ = LayerType::kLayerUnknown;
  base::DataType data_type_ = base::DataType::kDataTypeUnknown;
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
};

class Layer : public BaseLayer {
 public:
  explicit Layer(base::DeviceType device_type, LayerType layer_type,
                 std::string layer_name = "");

  base::Status forward() override;

  base::Status forward(const std::vector<tensor::Tensor>& inputs,
                       std::vector<tensor::Tensor>& outputs) override;

  void set_input(int32_t idx, const tensor::Tensor& input) override;

  void set_output(int32_t idx, const tensor::Tensor& output) override;

  size_t input_size() const override;

  size_t output_size() const override;

  base::Status check() const override;

  tensor::Tensor& get_input(int32_t idx) override;

  tensor::Tensor& get_output(int32_t idx) override;

  const tensor::Tensor& get_input(int32_t idx) const override;

  const tensor::Tensor& get_output(int32_t idx) const override;

  base::Status check_tensor(const tensor::Tensor& tensor,
                            base::DeviceType device_type,
                            base::DataType data_type) const;

  base::Status check_tensor_with_dim(const tensor::Tensor& tensor,
                                     base::DeviceType device_type,
                                     base::DataType data_type,
                                     std::initializer_list<int32_t> dims) const;

  void reset_input_size(size_t size);

  void reset_output_size(size_t size);

  virtual void to_cuda();

  void set_cuda_config(std::shared_ptr<base::CudaConfig> config);

  std::shared_ptr<base::CudaConfig> cuda_config() const;

 protected:
  std::vector<tensor::Tensor> inputs_;
  std::vector<tensor::Tensor> outputs_;
  std::shared_ptr<base::CudaConfig> cuda_config_;
};

class LayerParam : public Layer {
 public:
  explicit LayerParam(base::DeviceType device_type, LayerType layer_type,
                      bool is_quant_layer = false, std::string layer_name = "");

  size_t weight_size() const;

  void reset_weight_size(size_t size);

  tensor::Tensor& get_weight(int32_t idx);

  const tensor::Tensor& get_weight(int32_t idx) const;

  void to_cuda() override;

  base::Status set_weight(int32_t idx, const tensor::Tensor& weight) override;

  base::Status set_weight(
      int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
      base::DeviceType device_type = base::DeviceType::kDeviceUnknown) override;

  void set_scales(const tensor::Tensor& scales);

  void set_group_size(int32_t group_size);

  int32_t get_scale_num() const;

 protected:
  int32_t group_size_ = 0;
  bool is_quant_layer_ = false;
  tensor::Tensor scales_;
  std::vector<tensor::Tensor> weights_;
};
}  // namespace op

#endif  // MICROLLM_INCLUDE_OP_LAYER_H
