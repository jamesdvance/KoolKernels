#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define TILE_WIDTH 16

// tiled matmul
__global__ void tiledMatMul(float* A, float* B, float* C, int widthA, int heightA, int widthB) {

    __shared__ float tileA; //trying as row-major. Would be TILE_WIDTH * TILE_WIDTH
    __shared__ float tileB; 

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // global row and column output of C to load into memory
    // 
    int rowC = blockIdx.y * TILE_WIDTH + blockIdx.y; 
    int colC = blockIdx.x * TILE_WIDTH + blockIdx.x; 

    // we iterate over how many N tiles it takes to cover the common dimension. CAlling the iterator ph for 'phase'
    for (int ph = 0; ph < TILE_WIDTH/widthA, ph++){
        
        // Check if we should load into tileA for this phase
        // Check conditions - 
        if (ph * TILE_WIDTH < widthA  ){
            tileA[TILE_WIDTH] = A[ph*TILE_WIDTH * ty ];

        } else{
            tileA[tx*ty] = 0f;
        }




    }


};


// Wrapper functions - Pytorch

torch::Tensor matmul_cuda_forward(torch::Tensor A, torch::Tensor B) {
    const int heightA = A.size(0);
    const int widthA = A.size(1);
    const int heightB = B.size(0);
    const int widthB = B.size(1);
    
    auto C = torch::zeros({heightA, widthB}, A.options());
    
    const int threadsPerBlockX = 16;
    const int threadsPerBlockY = 16;
    const dim3 threadsPerBlock(threadsPerBlockX, threadsPerBlockY);
    const dim3 numBlocks((widthB + threadsPerBlockX - 1) / threadsPerBlockX,
                         (heightA + threadsPerBlockY - 1) / threadsPerBlockY);
    
    
    tiledMatMul<<<numBlocks, threadsPerBlock>>>(
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