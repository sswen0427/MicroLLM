#include "base/base.h"

#include <iostream>

namespace base {

Status::Status(int code, std::string msg) {}

Status::operator bool() const { return true; }

int Status::get_err_code() const { return 0; }

}  // namespace base
