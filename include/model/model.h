#ifndef MICROLLM_INCLUDE_MODEL_MODEL_H
#define MICROLLM_INCLUDE_MODEL_MODEL_H
#include <string>

class Model {
 public:
  explicit Model(std::string token_path, std::string model_path);

  virtual void Init() = 0;

 private:
  std::string token_path_;
};

#endif  // MICROLLM_INCLUDE_MODEL_MODEL_H
