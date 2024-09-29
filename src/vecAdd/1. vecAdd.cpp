#include <vector> 
#include <string> 
#include <cuda_runtime.h>
#include "readCSV.cpp"


// CUDA kernel for vector addition
__global__ void vecAdd(float* A, float* B, float* C, int n){
    /* Row-major addition of a kernel
        threadIdx: 
        blockDim: 
        blockIdx: 
    */
    int i = threadIdx.x + blockDim.x * blockIdx.x; 
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}


int main(){
    std::string file1 = "data/df1_100.csv";
    std::string file2 = "data/df2_100.csv";

    std::vector<double> h_vector1 = readCSV(file1);
    std::vector<double> h_vector2 = readCSV(file2);

    if(h_vector1.size() != h_vector2.size()){
        std::cerr << "Error: Vectors have different sizes" << std::endl; 
        return 1; 
    }

    int numElements = h_vector1.size();
    size_t size = numElements * sizeof(double);

    // Allocate device memory
    double *d_vector1, *d_vector2, *d_result;
    cudaMalloc((void**)&d_vector1, size); 
    cudaMalloc((void**)&d_vector2, size); 
    cudaMalloc((void**)&d_result, size); 

    // Copy host vectors to device
    cudaMemcpy(d_vector1, h_vector1.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector1, h_vector2.data(), size, cudaMemcpyHostToDevice);

    // Launch Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock -1)/ threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_vector1, d_vector2, d_result, numElements);
    




}