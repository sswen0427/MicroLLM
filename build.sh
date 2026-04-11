#!/bin/bash
set -e
set -x

show_help() {
    echo "Usage: ./build.sh [MODE]"
    echo ""
    echo "Modes:"
    echo "  debug    : Compile in Debug mode (-g -G, injects CUDA_DEBUG macro). Best for development."
    echo "  release  : Compile in Release mode (-O3 optimization). Best for performance testing."
    echo "  help     : Show this help message."
    echo ""
}
# Default to 'release' mode if no argument is provided
MODE=${1:-"release"}

# Get the number of CPU cores for parallel compilation
CORES=$(nproc 2>/dev/null || echo 4)

case $MODE in
    "debug")
        echo "[Debug Mode] Preparing build environment..."

        # Force clean to prevent CMake cache pollution
        rm -rf build
        mkdir build
        cd build

        # Abstract CMake arguments into an array
        CMAKE_ARGS=(
            "-DCMAKE_BUILD_TYPE=Debug"
            "-DCMAKE_CXX_FLAGS=-DCUDA_DEBUG"
            "-DCMAKE_CUDA_FLAGS=-DCUDA_DEBUG"
        )

        echo "[Debug Mode] Configuring CMake..."
        cmake .. "${CMAKE_ARGS[@]}"

        echo "[Debug Mode] Starting multi-core compilation (${CORES} cores)..."
        make -j${CORES}
        echo "Debug build completed successfully!"
        ;;

    "release")
        echo "[Release Mode] Preparing build environment..."

        rm -rf build
        mkdir build
        cd build

        CMAKE_ARGS=(
            "-DCMAKE_BUILD_TYPE=Release"
            "-DCMAKE_CXX_FLAGS=-DNDEBUG"
            "-DCMAKE_CUDA_FLAGS=-DNDEBUG"
        )

        echo "[Release Mode] Configuring CMake..."
        cmake .. "${CMAKE_ARGS[@]}"

        echo "[Release Mode] Starting multi-core compilation (${CORES} cores)..."
        make -j${CORES}
        echo "Release build completed successfully!"
        ;;

    "help"|"-h"|"--help")
        show_help
        exit 0
        ;;

    *)
        echo "Error: Unknown mode '$MODE'"
        show_help
        exit 1
        ;;
esac