#include <gtest/gtest.h>
#include "base/alloc.h"
#include "base/buffer.h"


TEST(BufferTest, Allocate) {
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    base::Buffer buffer(32, alloc);
    EXPECT_NE(buffer.ptr(), nullptr);
}
