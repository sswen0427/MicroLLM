cmake_minimum_required(VERSION 3.22)
project(MicroLLM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
list(APPEND CMAKE_PREFIX_PATH
        "${THIRD_PARTY_DIR}/gflags"
        "${THIRD_PARTY_DIR}/glog"
        "${THIRD_PARTY_DIR}/gtest"
        "${THIRD_PARTY_DIR}/unwind"
)

find_package(gflags REQUIRED)
message(STATUS "Found gflags: ${gflags_DIR}")

find_package(glog REQUIRED)
message(STATUS "Found glog: ${glog_DIR}")


add_executable(MicroLLM main.cpp)
target_link_libraries(MicroLLM PRIVATE glog::glog gflags_shared)
