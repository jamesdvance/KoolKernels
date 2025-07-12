#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include matMul.cu

// Forward declaration of the CUDA kernel from matMul.cu
__global__ void matMulSimplest(float* A, float* B, float* C, int widthA, int heightA, int widthB);

torch::Tensor matmul_cuda_forward(torch::Tensor A, torch::Tensor B) {
    const int heightA = A.size(0);
    const int widthA = A.size(1);
    const int heightB = B.size(0);
    const int widthB = B.size(1);
    
    TORCH_CHECK(widthA == heightB, "Matrix dimensions must be compatible for multiplication");
    TORCH_CHECK(A.is_contiguous(), "Tensor A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "Tensor B must be contiguous");
    
    auto C = torch::zeros({heightA, widthB}, A.options());
    
    const int threadsPerBlockX = 16;
    const int threadsPerBlockY = 16;
    const dim3 threadsPerBlock(threadsPerBlockX, threadsPerBlockY);
    const dim3 numBlocks((widthB + threadsPerBlockX - 1) / threadsPerBlockX,
                         (heightA + threadsPerBlockY - 1) / threadsPerBlockY);
    
    matMulSimplest<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        widthA, heightA, widthB
    );
    
    cudaDeviceSynchronize();
    return C;
}

torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B) {
    if (A.is_cuda() && B.is_cuda()) {
        return matmul_cuda_forward(A, B);
    } else {
        return torch::mm(A, B);  // Fallback to PyTorch's implementation
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_forward", &matmul_forward, "Matrix multiplication forward pass");
}