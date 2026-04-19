#pragma once

#include "base/base.h"
#include "op/layer.h"

namespace op {
/**
 * @brief Matrix multiplication layer (also known as Linear or Dense layer).
 * Computes the operation: Y = X * W + B (if bias is enabled).
 */
class MatmulLayer : public LayerParam {
 public:
  /**
   * @brief Constructs a MatmulLayer instance.
   * @param device_type The device where the layer will run (e.g., CPU, CUDA).
   * @param dim0 The first dimension of the weight matrix (e.g., in_features or
   * K).
   * @param dim1 The second dimension of the weight matrix (e.g., out_features
   * or N).
   * @param is_quant_layer Flag indicating if the layer uses quantized weights
   * (e.g., INT8/INT4).
   * @param has_bias Flag indicating if a bias term should be added after
   * multiplication.
   */
  explicit MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1,
                       bool is_quant_layer = false, bool has_bias = false);

  /**
   * @brief Validates the layer's parameters, weights, and input/output tensor
   * shapes.
   * @return Status indicating success or specific error during validation.
   */
  [[nodiscard]] base::Status check() const override;

  /**
   * @brief Executes the forward pass of the matrix multiplication.
   * @return Status indicating success or runtime error during execution.
   */
  base::Status forward() override;

  /**
   * @brief Initializes and sets the bias tensor data.
   * @param idx The index of the bias tensor in the internal container.
   * @param dim The dimension size of the bias tensor (usually equals dim1).
   * @param bias_ptr Raw pointer to the bias data.
   * @param device_type The memory location of the provided bias_ptr (CPU or
   * CUDA).
   * @return Status indicating if the bias was set successfully.
   */
  base::Status set_bias(int32_t idx, int32_t dim, const void* bias_ptr,
                        base::DeviceType device_type);

  /**
   * @brief Retrieves a reference to the bias tensor.
   * @param idx The index of the bias tensor.
   * @return Modifiable reference to the bias Tensor.
   */
  tensor::Tensor& get_bias(int32_t idx);

  /**
   * @brief Retrieves a constant reference to the bias tensor.
   * @param idx The index of the bias tensor.
   * @return Read-only reference to the bias Tensor.
   */
  const tensor::Tensor& get_bias(int32_t idx) const;

  /**
   * @brief Migrates the layer's internal parameters (e.g., bias) to GPU memory.
   */
  void to_cuda() override;

 private:
  /// @brief First dimension of the weight matrix (in_features).
  int32_t dim0_ = 0;

  /// @brief Second dimension of the weight matrix (out_features).
  int32_t dim1_ = 0;

  /// @brief Indicates whether this layer applies a bias addition.
  bool has_bias_ = false;

  /// @brief Container storing the bias tensor(s).
  /// Uses a vector to support potential multi-head or grouped biases.
  std::vector<tensor::Tensor> bias_;
};
}  // namespace op
