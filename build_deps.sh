#!/bin/bash

set -e
set -x

rm -rf third_party
git clone https://github.com/sswen0427/cpp3rdlib.git
mv cpp3rdlib third_party
cd third_party
rm .git -rf
mv abseil-cpp-20260107.1-debian12-x86_64-gcc12.2.0 abseil-cpp
mv armadillo-15.2.4-debian12-x86_64-gcc12.2.0 armadillo
mv boost-1.90.0-debian12-x86_64-gcc12.2.0 boost
mv gflags-2.2.2-debian12-x86_64-gcc12.2.0 gflags
mv glog-0.7.1-debian12-x86_64-gcc12.2.0 glog
mv gtest-1.17.0-debian12-x86_64-gcc12.2.0 gtest
mv openblas-0.3.32-debian12-x86_64-gcc12.2.0 openblas
mv re2-2025-11-05-debian12-x86_64-gcc12.2.0 re2
mv sentencepiece-0.2.1-debian12-x86_64-gcc12.2.0 sentencepiece
mv unordered_dense-4.8.1-debian12-x86_64-gcc12.2.0 unordered_dense
mv unwind-1.8.3-debian12-x86_64-gcc12.2.0 unwind
