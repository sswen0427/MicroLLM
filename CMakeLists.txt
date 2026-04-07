# ==============================================================================
# Project Base Configuration
# ==============================================================================
# Require a minimum CMake version of 3.22 for modern C++ and CUDA support.
cmake_minimum_required(VERSION 3.22)

# [Critical Configuration]: Manually specify the target CUDA architecture.
# '80' stands for Ampere architecture (e.g., A100/A800).
# This prevents CMake from failing when probing for a GPU on a machine without a physical GPU.
set(CMAKE_CUDA_ARCHITECTURES 80)

# [Critical Configuration]: Explicitly set the absolute path to the nvcc compiler.
# This prevents "compiler not found" errors caused by IDEs (like CLion) not reading ~/.bashrc.
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.1/bin/nvcc")

# Define the project name and enable both C++ (CXX) and CUDA language support.
project(MicroLLM LANGUAGES CXX CUDA)

# Enforce C++20 standard for modern C++ features (e.g., Concepts, Modules).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==============================================================================
# Third-party Libraries Configuration
# ==============================================================================
# Define the root directory for local third-party libraries.
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

# Append local third-party library paths to CMAKE_PREFIX_PATH
# so that `find_package` can locate their corresponding xxxConfig.cmake files.
list(APPEND CMAKE_PREFIX_PATH
        "${THIRD_PARTY_DIR}/gflags"
        "${THIRD_PARTY_DIR}/glog"
        "${THIRD_PARTY_DIR}/gtest"
        "${THIRD_PARTY_DIR}/unwind"
        "${THIRD_PARTY_DIR}/sentencepiece"
)

# Find and load Google Flags (gflags) for command-line argument parsing.
find_package(gflags REQUIRED)
message(STATUS "Found gflags: ${gflags_DIR}")

# Find and load Google Logging (glog) for high-performance logging.
find_package(glog REQUIRED)
message(STATUS "Found glog: ${glog_DIR}")

# Find and load the CUDA Toolkit (provides CUDA runtime libraries and headers).
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDAToolkit: ${CUDAToolkit_DIR}")

# Manually configure SentencePiece (tokenizer) include and library directories.
set(SENTENCEPIECE_ROOT "${THIRD_PARTY_DIR}/sentencepiece")
set(SENTENCEPIECE_INCLUDE_DIR "${SENTENCEPIECE_ROOT}/include")
set(SENTENCEPIECE_LIB_DIR "${SENTENCEPIECE_ROOT}/lib")

# ==============================================================================
# Source & Header Files Collection
# ==============================================================================
# Recursively search for all subdirectories under include/ and add them to ALL_HEADER_DIRS.
# This allows for shorter and cleaner #include paths in the source code.
file(GLOB_RECURSE HEADER_DIRS_FILES LIST_DIRECTORIES true CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*")
set(ALL_HEADER_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
foreach(path IN LISTS HEADER_DIRS_FILES)
    if(IS_DIRECTORY "${path}")
        list(APPEND ALL_HEADER_DIRS "${path}")
    endif()
endforeach()

# Recursively collect all source files (.cpp, .cu, etc.) under the src/ directory.
file(GLOB_RECURSE ALL_SRC_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*")

# ==============================================================================
# Build Target Definition
# ==============================================================================
# Define the final executable target 'MicroLLM' and compile it with main.cpp and all collected source files.
add_executable(MicroLLM main.cpp ${ALL_SRC_FILES})
# Add header search paths for the MicroLLM target.
# (PRIVATE means these paths are only visible to this target and not exported).
target_include_directories(MicroLLM PRIVATE ${ALL_HEADER_DIRS})
target_include_directories(MicroLLM PRIVATE ${SENTENCEPIECE_INCLUDE_DIR})
# [Fix]: Add CUDA Toolkit include directories to resolve '<cuda_runtime.h> not found' errors.
target_include_directories(MicroLLM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
# Add library search paths (tells the linker where to find .a or .so files).
target_link_directories(MicroLLM PRIVATE ${SENTENCEPIECE_LIB_DIR})
# Link all required third-party libraries to the final executable.
target_link_libraries(MicroLLM PRIVATE glog::glog sentencepiece CUDA::cudart)


enable_testing()
find_package(GTest REQUIRED)
message(STATUS "Found GTest: ${GTest_DIR}")

file(GLOB ALL_TEST_FILES "test/test_*.cpp")
add_executable(MicroLLM_test ${ALL_TEST_FILES} ${ALL_SRC_FILES})
target_include_directories(MicroLLM_test PRIVATE ${ALL_HEADER_DIRS})
target_include_directories(MicroLLM_test PRIVATE ${SENTENCEPIECE_INCLUDE_DIR})
target_include_directories(MicroLLM_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_directories(MicroLLM_test PRIVATE ${SENTENCEPIECE_LIB_DIR})
target_link_libraries(MicroLLM_test PRIVATE glog::glog sentencepiece CUDA::cudart GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(MicroLLM_test)
