# ==============================================================================
# Project Base Configuration
# ==============================================================================
# Require a minimum CMake version of 3.22 for modern C++ and CUDA support.
cmake_minimum_required(VERSION 3.22)

# [Critical Configuration]: Manually specify the target CUDA architecture.
# '80' stands for Ampere architecture (e.g., A100/A800).
# This prevents CMake from failing when probing for a GPU on a machine without a physical GPU.
set(CMAKE_CUDA_ARCHITECTURES 80)

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
        "${THIRD_PARTY_DIR}/abseil-cpp"
        "${THIRD_PARTY_DIR}/armadillo"
        "${THIRD_PARTY_DIR}/boost"
        "${THIRD_PARTY_DIR}/gflags"
        "${THIRD_PARTY_DIR}/glog"
        "${THIRD_PARTY_DIR}/gtest"
        "${THIRD_PARTY_DIR}/json"
        "${THIRD_PARTY_DIR}/openblas"
        "${THIRD_PARTY_DIR}/re2"
        "${THIRD_PARTY_DIR}/sentencepiece"
        "${THIRD_PARTY_DIR}/unordered_dense"
        "${THIRD_PARTY_DIR}/unwind"
)

find_package(absl REQUIRED)
find_package(Armadillo REQUIRED CONFIG)
find_package(Boost REQUIRED CONFIG)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(GTest REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(re2 REQUIRED)
find_package(sentencepiece REQUIRED)
find_package(unordered_dense CONFIG REQUIRED)
find_package(Unwind REQUIRED)
find_package(CUDAToolkit REQUIRED)

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
# [Fix]: Add CUDA Toolkit include directories to resolve '<cuda_runtime.h> not found' errors.
target_include_directories(MicroLLM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
# Link all required third-party libraries to the final executable.
target_link_libraries(MicroLLM PRIVATE glog::glog sentencepiece::sentencepiece CUDA::cudart Boost::boost unordered_dense::unordered_dense re2::re2 nlohmann_json::nlohmann_json)


enable_testing()
file(GLOB_RECURSE ALL_TEST_FILES CONFIGURE_DEPENDS "test/test_*.cpp")

add_executable(MicroLLM_test ${ALL_TEST_FILES} ${ALL_SRC_FILES})
target_include_directories(MicroLLM_test PRIVATE ${ALL_HEADER_DIRS})
target_include_directories(MicroLLM_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(MicroLLM_test PRIVATE glog::glog sentencepiece::sentencepiece CUDA::cudart Boost::boost GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(MicroLLM_test)
