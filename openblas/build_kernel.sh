#!/usr/bin/bash
set -eo pipefail

BUILD_DIR="build"

if [ ! command -v cmake &> /dev/null ]; then
    echo "cmake could not be found. Please install cmake and try again."
    echo "Fedora: sudo dnf install cmake"
    echo "Ubuntu: sudo apt install cmake"
    exit 1
fi

echo "== Setting up cmake in $BUILD_DIR"

cmake -S ./kernel_src -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release 

echo "== Building OpenBLAS kernel"
cmake --build "$BUILD_DIR" --target all -- -j $(nproc)

echo "== Build complete. You can run the kernel through $BUILD_DIR/kernel"
