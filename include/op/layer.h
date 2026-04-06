#ifndef MICROLLM_INCLUDE_OP_LAYER_H
#define MICROLLM_INCLUDE_OP_LAYER_H
#include <string>

#include "base/base.h"
#include "tensor/tensor.h"

namespace op {
enum class LayerType : uint8_t {
  kLayerUnknown = 0,
  kLayerLinear = 1,
};

enum class LayerStatus : uint8_t {
  kForwardSuccess = 0,
  kFunctionUnImplement = 1,
};

class Layer {
 public:
  explicit Layer(LayerType layer_type, base::DataType data_type,
                 std::string layer_name = "");

  base::DataType data_type() const;

  LayerType layer_type() const;

  virtual LayerStatus Init() = 0;

  virtual LayerStatus Forward() = 0;

  virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

  virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

  virtual tensor::Tensor get_input(int32_t idx) const = 0;

  virtual tensor::Tensor get_output(int32_t idx) const = 0;

  virtual void set_weight(int32_t idx, const tensor::Tensor& weight) = 0;

  virtual tensor::Tensor get_weight(int32_t idx) const = 0;

  virtual void reset_input_size(size_t size) = 0;

  virtual void reset_output_size(size_t size) = 0;

 private:
  std::string layer_name_;

  base::DataType data_type_ = base::DataType::kDataTypeUnknown;

  LayerType layer_type_ = LayerType::kLayerUnknown;
};

class ParamLayerFp32 : public Layer {
 public:
  explicit ParamLayerFp32(LayerType layer_type, std::string layer_name = "");

  LayerStatus Init() override;

  LayerStatus Forward() override;

  void set_input(int32_t idx, const tensor::Tensor& input) override;

  void set_output(int32_t idx, const tensor::Tensor& output) override;

  tensor::Tensor get_input(int32_t idx) const override;

  tensor::Tensor get_output(int32_t idx) const override;

  void set_weight(int32_t idx, const tensor::Tensor& weight) override;

  tensor::Tensor get_weight(int32_t idx) const override;

  void reset_weight_size(size_t size);

  void reset_input_size(size_t size) override;

  void reset_output_size(size_t size) override;

 private:
  std::vector<tensor::Tensor> weights_;

  std::vector<tensor::Tensor> inputs_;

  std::vector<tensor::Tensor> outputs_;
};

}  // namespace op

#endif  // MICROLLM_INCLUDE_OP_LAYER_H
