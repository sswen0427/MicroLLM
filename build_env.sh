# Install cuda without driver
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
sudo sh cuda_12.8.0_570.86.10_linux.run

export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Install cmake
cd /tmp
wget https://github.com/Kitware/CMake/releases/download/v4.3.1/cmake-4.3.1-linux-x86_64.tar.gz
tar -zxvf cmake-4.3.1-linux-x86_64.tar.gz
sudo mv cmake-4.3.1-linux-x86_64 /usr/local/cmake
export PATH=/usr/local/cmake/bin:$PATH