#!/bin/bash

set -e
set -x

mkdir third_party && cd third_party

# Build glog library, which depends on gflags and libunwind
# Build gflags library
git clone https://github.com/gflags/gflags.git
cd gflags
# See https://github.com/google/glog/blob/v0.7.1/CMakeLists.txt#L77
# glog needs gflags 2.2.2 version
git checkout v2.2.2
mkdir build && cd build
cmake \
  -DCMAKE_INSTALL_PREFIX=gflags_install \
  -DBUILD_SHARED_LIBS=ON \
  ..
make -j$(nproc)
make install
mv gflags_install ../../
cd ../..
rm -rf gflags
mv gflags_install gflags


# Build libunwind library
sudo apt-get update
sudo apt-get install autoconf automake libtool pkg-config
git clone https://github.com/libunwind/libunwind.git
cd libunwind
git checkout v1.8.3
autoreconf -i -f
INSTALL_DIR="$(pwd)/../unwind"
./configure --prefix="${INSTALL_DIR}" --enable-shared
make -j$(nproc)
make install
cd ..
rm -rf libunwind

# Build glog library
git clone https://github.com/google/glog.git
cd glog
git checkout v0.7.1
mkdir build && cd build
cmake \
  -DCMAKE_INSTALL_PREFIX=glog_install \
  -DCMAKE_PREFIX_PATH=../../gflags \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTING=OFF \
  ..
make -j$(nproc)
make install
mv glog_install ../../
cd ../..
rm -rf glog
mv glog_install glog

# Build googletest(gtest) library
git clone https://github.com/google/googletest.git
cd googletest
git checkout v1.17.0
mkdir build && cd build
cmake \
  -DCMAKE_INSTALL_PREFIX=gtest_install \
  -DBUILD_SHARED_LIBS=ON \
  ..
make -j$(nproc)
make install
mv gtest_install ../../
cd ../..
rm -rf googletest
mv gtest_install gtest

git clone https://github.com/google/sentencepiece.git
cd sentencepiece
git checkout v0.2.1
mkdir build && cd build
cmake \
  -DCMAKE_INSTALL_PREFIX=sentencepiece_install \
  -DBUILD_SHARED_LIBS=ON \
  ..
make -j$(nproc)
make install
mv sentencepiece_install ../../
cd ../..
rm -rf sentencepiece
mv sentencepiece_install sentencepiece
