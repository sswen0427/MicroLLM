#ifndef MICROLLM_INCLUDE_MODEL_MODEL_H
#define MICROLLM_INCLUDE_MODEL_MODEL_H
#include <string>
#include <vector>

#include "base.h"
#include "tensor.h"
#include

class Model {
 public:
  explicit Model(ModelType model_type, std::string token_path,
                 std::string model_path);

  virtual Status init() = 0;

  virtual Tensor forward(const std::vector<int>& tokens, int start_pos) = 0;

  ModelType model_type() const;

  const std::string& token_path() const;

  const std::string& model_path() const;

 private:
  ModelType model_type_;
  std::string token_path_;
  std::string model_path_;
  std::unique_ptr<sentence>
};

#endif  // MICROLLM_INCLUDE_MODEL_MODEL_H
