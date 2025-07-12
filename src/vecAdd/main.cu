#include <vector> 
#include <string> 
#include <cuda_runtime.h>
#include "../readCSV.cpp"

// Forward declaration of CUDA kernel
__global__ void vecAdd(float* A, float* B, float* C, int n);

int main(){
    std::string file1 = "../data/df_1_x_100_int.csv";
    std::string file2 = "../data/df_1_x_100_int.csv";

    std::vector<double> h_vector1_double = readCSV(file1);
    std::vector<double> h_vector2_double = readCSV(file1);
    
    // Convert to float vectors for CUDA kernel
    std::vector<float> h_vector1(h_vector1_double.begin(), h_vector1_double.end());
    std::vector<float> h_vector2(h_vector2_double.begin(), h_vector2_double.end());

    if(h_vector1.size() != h_vector2.size()){
        std::cerr << "Error: Vectors have different sizes" << std::endl; 
        return 1;
    }

    int numElements = h_vector1.size();
    size_t size = numElements * sizeof(float);

    // Allocate device memory
    float *d_vector1, *d_vector2, *d_result;
    cudaMalloc((void**)&d_vector1, size); 
    cudaMalloc((void**)&d_vector2, size); 
    cudaMalloc((void**)&d_result, size); 

    // Copy host vectors to device
    cudaMemcpy(d_vector1, h_vector1.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector2, h_vector2.data(), size, cudaMemcpyHostToDevice);

    // Launch Kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElements + threadsPerBlock -1)/ threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_vector1, d_vector2, d_result, numElements);
    std::out << "Vector Addition Completed Successfully" << std::endl;
    // TODO - add memC
    std::vector<float> output_vector(numElements);
    cudaMemcpy(output_vector.data(), d_result, size, cudaMemcpyDeviceToHost);

}