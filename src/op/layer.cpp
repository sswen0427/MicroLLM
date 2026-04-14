#include "layer.h"

#include <numeric>

#include "base/base.h"
#include "glog/logging.h"
#include "tensor/tensor.h"

// BaseLayer
namespace op {
BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type,
                     base::DataType data_type, std::string layer_name)
    : device_type_(device_type),
      layer_type_(layer_type),
      data_type_(data_type),
      layer_name_(std::move(layer_name)) {}

base::DataType BaseLayer::data_type() const { return data_type_; }

LayerType BaseLayer::layer_type() const { return layer_type_; }

const std::string& BaseLayer::get_layer_name() const { return layer_name_; }

void BaseLayer::set_layer_name(const std::string& layer_name) {
  layer_name_ = layer_name;
}
base::DeviceType BaseLayer::device_type() const { return device_type_; }

void BaseLayer::set_device_type(base::DeviceType device_type) {
  device_type_ = device_type;
}
}  // namespace op

// Layer
namespace op {
Layer::Layer(base::DeviceType device_type, LayerType layer_type,
             std::string layer_name)
    : BaseLayer(device_type, layer_type, base::DataType::kDataTypeFp32,
                std::move(layer_name)) {}

base::Status Layer::forward() { return base::error::FunctionNotImplement(""); }

base::Status Layer::forward(const std::vector<tensor::Tensor>& inputs,
                            std::vector<tensor::Tensor>& outputs) {
  for (const auto& input : inputs) {
    inputs_.emplace_back(input);
  }
  for (const auto& output : outputs) {
    outputs_.emplace_back(output);
  }
  return this->forward();
}
void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  this->inputs_.at(idx) = input;
}

void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  this->outputs_.at(idx) = output;
}

size_t Layer::input_size() const { return inputs_.size(); }

size_t Layer::output_size() const { return outputs_.size(); }
base::Status Layer::check() const {
  return base::error::FunctionNotImplement(
      "The check function is not implement yet");
}
tensor::Tensor& Layer::get_input(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

tensor::Tensor& Layer::get_output(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

const tensor::Tensor& Layer::get_input(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

const tensor::Tensor& Layer::get_output(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

base::Status Layer::check_tensor(const tensor::Tensor& tensor,
                                 base::DeviceType device_type,
                                 base::DataType data_type) const {
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }
  return base::error::Success();
}

base::Status Layer::check_tensor_with_dim(
    const tensor::Tensor& tensor, base::DeviceType device_type,
    base::DataType data_type, std::initializer_list<int32_t> dims) const {
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }
  if (tensor.dims_size() != static_cast<int32_t>(dims.size())) {
    return base::error::InvalidArgument(
        "The tensor dimension count mismatch. Expected: " +
        std::to_string(dims.size()) +
        ", Got: " + std::to_string(tensor.dims_size()));
  }
  int32_t i = 0;
  for (int32_t expected_dim : dims) {
    if (tensor.get_dim(i) != expected_dim) {
      return base::error::InvalidArgument(
          "The tensor has a wrong dim at index " + std::to_string(i) +
          ". Expected: " + std::to_string(expected_dim) +
          ", Got: " + std::to_string(tensor.get_dim(i)));
    }
    ++i;
  }
  return base::error::Success();
}

void Layer::reset_input_size(size_t size) { inputs_.resize(size); }

void Layer::reset_output_size(size_t size) { outputs_.resize(size); }

void Layer::to_cuda() {
  for (auto& input : inputs_) {
    if (!input.is_empty()) {
      input.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
  for (auto& output : outputs_) {
    if (!output.is_empty()) {
      output.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
}

void Layer::set_cuda_config(std::shared_ptr<base::CudaConfig> config) {
  if (!config) {
    return;
  }
  this->cuda_config_ = config;
}

std::shared_ptr<base::CudaConfig> Layer::cuda_config() const {
  return cuda_config_;
}
}  // namespace op

// LayerParam
namespace op {
LayerParam::LayerParam(base::DeviceType device_type, LayerType layer_type,
                       bool is_quant_layer, std::string layer_name)
    : Layer(device_type, layer_type, std::move(layer_name)),
      is_quant_layer_(is_quant_layer) {}

void LayerParam::to_cuda() {
  Layer::to_cuda();
  for (auto& weight : weights_) {
    weight.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
  }
  if (!scales_.is_empty()) {
    scales_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
  }
}

base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
  if (!weight.is_empty()) {
    CHECK(weight.device_type() == device_type_);
  }
  weights_.at(idx) = weight;
  return base::error::Success();
}

base::Status LayerParam::set_weight(int32_t idx,
                                    const std::vector<int32_t>& dims,
                                    const void* weight_ptr,
                                    base::DeviceType device_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK_NE(weight_ptr, nullptr);

  size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(float),
                                std::multiplies<>());
  std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(
      size, nullptr, const_cast<void*>(weight_ptr));
  if (device_type != base::DeviceType::kDeviceUnknown) {
    buffer->set_device_type(device_type);
  }

  if (!is_quant_layer_) {
    tensor::Tensor weight = tensor::Tensor::from_external(
        base::DataType::kDataTypeFp32, dims, buffer.get());
    weight.set_device_type(device_type);
    weights_.at(idx) = weight;
  } else {
    // is quant layer
    tensor::Tensor weight = tensor::Tensor::from_external(
        base::DataType::kDataTypeInt8, dims, buffer.get());
    weight.set_device_type(device_type);
    weights_.at(idx) = weight;

    const int32_t weight_size = static_cast<int32_t>(weight.size());
    CHECK(weight_size % group_size_ == 0);

    int32_t scale_nums = weight_size / group_size_;
    scales_ = tensor::Tensor::from_external(base::DataType::kDataTypeFp32,
                                            {scale_nums},
                                            (int8_t*)weight_ptr + weight_size);
    scales_.set_device_type(device_type);
  }

  return base::error::Success();
}

size_t LayerParam::weight_size() const { return weights_.size(); }

const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

int32_t LayerParam::get_scale_num() const {
  CHECK(!scales_.is_empty());
  return static_cast<int32_t>(scales_.size());
}

void LayerParam::reset_weight_size(size_t size) { weights_.resize(size); }

tensor::Tensor& LayerParam::get_weight(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

void LayerParam::set_scales(const tensor::Tensor& scales) {
  CHECK(!scales.is_empty());
  this->scales_ = scales;
}

void LayerParam::set_group_size(int32_t group_size) {
  this->group_size_ = group_size;
}

}  // namespace op