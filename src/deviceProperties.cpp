#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Maximum threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Maxmimum threads in X-dimension of block: " << prop.maxThreadsDim[0] << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl; 

    return 0;
}