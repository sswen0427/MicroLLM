#!/bin/bash

set -e
set -x

rm -rf third_party
git clone https://github.com/sswen0427/cpp3rdlib.git
mv cpp3rdlib third_party
cd third_party
rm .git -rf
mv armadillo-15.2.4-debian12-x86_64-gcc12.2.0 armadillo
mv gflags-2.2.2-debian12-x86_64-gcc12.2.0 gflags
mv gtest-1.17.0-debian12-x86_64-gcc12.2.0 gtest
mv sentencepiece-0.2.1-debian12-x86_64-gcc12.2.0 sentencepiece
mv boost-1.90.0-debian12-x86_64-gcc12.2.0 boost
mv glog-0.7.1-debian12-x86_64-gcc12.2.0 glog
mv openblas-0.3.32-debian12-x86_64-gcc12.2.0 openblas
mv unwind-1.8.3-debian12-x86_64-gcc12.2.0 unwind
