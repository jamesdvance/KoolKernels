# Create simple CUDA test
echo '
#include <cuda_runtime.h>
#include <iostream>
int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA devices: " << deviceCount << std::endl;
    return 0;
}' > cuda_test.cu

# Compile and run (if nvcc is available)
nvcc cuda_test.cu -o cuda_test
./cuda_test