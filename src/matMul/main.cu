#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include "../readCSV.cpp"

// Forward declaration of CUDA kernel from matMul.cu
__global__ void matMul(float* A, float* B, float* C, int widthA, int heightA, int widthB);

int main() {
    std::string file1 = "../data/matrix1.csv";
    std::string file2 = "../data/matrix2.csv";
    
    // Read CSV files
    std::vector<double> h_matrix1_double = readCSV(file1);
    std::vector<double> h_matrix2_double = readCSV(file2);
    
    // Convert to float vectors for CUDA kernel
    std::vector<float> h_matrix1(h_matrix1_double.begin(), h_matrix1_double.end());
    std::vector<float> h_matrix2(h_matrix2_double.begin(), h_matrix2_double.end());
    
    // For this example, assuming square matrices - you may need to adjust dimensions
    int widthA = 100;  // Adjust based on your matrix dimensions
    int heightA = 100;
    int widthB = 100;
    int heightB = 100;
    
    if (h_matrix1.size() != widthA * heightA || h_matrix2.size() != widthB * heightB) {
        std::cerr << "Error: Matrix dimensions don't match expected size" << std::endl;
        return 1;
    }
    
    if (widthA != heightB) {
        std::cerr << "Error: Matrix dimensions incompatible for multiplication" << std::endl;
        return 1;
    }
    
    int widthC = widthB;
    int heightC = heightA;
    
    size_t sizeA = widthA * heightA * sizeof(float);
    size_t sizeB = widthB * heightB * sizeof(float);
    size_t sizeC = widthC * heightC * sizeof(float);
    
    // Allocate device memory
    float *d_matrix1, *d_matrix2, *d_result;
    cudaMalloc((void**)&d_matrix1, sizeA);
    cudaMalloc((void**)&d_matrix2, sizeB);
    cudaMalloc((void**)&d_result, sizeC);
    
    // Copy host matrices to device
    cudaMemcpy(d_matrix1, h_matrix1.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, h_matrix2.data(), sizeB, cudaMemcpyHostToDevice);
    
    // Launch kernel - using 16x16 thread blocks for matrix multiplication
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((widthC + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (heightC + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matMul<<<blocksPerGrid, threadsPerBlock>>>(d_matrix1, d_matrix2, d_result, widthA, heightA, widthB);
    
    // Copy result back to host
    std::vector<float> h_result(widthC * heightC);
    cudaMemcpy(h_result.data(), d_result, sizeC, cudaMemcpyDeviceToHost);
    
    std::cout << "Matrix multiplication completed successfully" << std::endl;
    
    // Free device memory
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_result);
    
    return 0;
}