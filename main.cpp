#include "alloc.h"
#include "tensor.h"

int main() {
  std::shared_ptr<CPUDeviceAllocator> allocator =
      std::make_shared<CPUDeviceAllocator>();
  Tensor tensor(DataType::kDataTypeFp32, 1, 2, 3, 4);
  tensor.allocate(allocator);
  tensor.allocate(allocator);

  tensor.reset(DataType::kDataTypeFp32, {4, 5, 6});
  tensor.allocate(allocator);

  tensor.reshape({11, 12, 13});
  const auto& strides = tensor.strides();
  return 0;
}
