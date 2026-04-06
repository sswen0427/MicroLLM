cmake_minimum_required(VERSION 3.22)
project(MicroLLM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
list(APPEND CMAKE_PREFIX_PATH
        "${THIRD_PARTY_DIR}/gflags"
        "${THIRD_PARTY_DIR}/glog"
        "${THIRD_PARTY_DIR}/gtest"
        "${THIRD_PARTY_DIR}/unwind"
        "${THIRD_PARTY_DIR}/sentencepiece"
)

find_package(gflags REQUIRED)
message(STATUS "Found gflags: ${gflags_DIR}")

find_package(glog REQUIRED)
message(STATUS "Found glog: ${glog_DIR}")

set(SENTENCEPIECE_ROOT "${THIRD_PARTY_DIR}/sentencepiece")
set(SENTENCEPIECE_INCLUDE_DIR "${SENTENCEPIECE_ROOT}/include")
set(SENTENCEPIECE_LIB_DIR "${SENTENCEPIECE_ROOT}/lib")

file(GLOB_RECURSE HEADER_DIRS_FILES LIST_DIRECTORIES true CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*")
set(ALL_HEADER_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
foreach(path IN LISTS HEADER_DIRS_FILES)
    if(IS_DIRECTORY "${path}")
        list(APPEND ALL_HEADER_DIRS "${path}")
    endif()
endforeach()


file(GLOB_RECURSE ALL_SRC_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*")

add_executable(MicroLLM main.cpp ${ALL_SRC_FILES})
target_include_directories(MicroLLM PRIVATE ${ALL_HEADER_DIRS})
target_include_directories(MicroLLM PRIVATE ${SENTENCEPIECE_INCLUDE_DIR})
target_link_directories(MicroLLM PRIVATE ${SENTENCEPIECE_LIB_DIR})
target_link_libraries(MicroLLM PRIVATE glog::glog sentencepiece)