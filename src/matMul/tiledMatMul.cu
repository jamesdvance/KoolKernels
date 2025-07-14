#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define TILE_WIDTH 16

// tiled matmul
__global__ void tiledMatMul(float* A, float* B, float* C, int widthA, int heightA, int widthB) {

    // Using T as a placeholder for specific type so we can handle float, double or __half
    // We can specify the type when launching the kernel like tiledMatMul<float><<<grid_dim, block_dim>>>()
    
    __shared__ T tileA[TILE_WIDTH][TILE_WIDTH]; 
    __shared__ T tileB[TILE_WIDTH][TILE_WIDTH]; 

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // global row / col
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    // global row and column output of C to load into memory
    // 
    int rowC = blockIdx.y * TILE_WIDTH + threadIdx.y; 
    int colC = blockIdx.x * TILE_WIDTH + threadIdx.x; 

    // we iterate over how many N tiles it takes to cover the common dimension. CAlling the iterator ph for 'phase'
    for (int ph = 0; ph <  widthA/TILE_WIDTH, ph++){
        
        // Check if we should load into tileA for this phase
        // Check conditions - 
        if ((ph * TILE_WIDTH * ty < widthA ) & ()){
            // index at 
            tileA[ty][tx] = A[TILE_WIDTH * ph*ty + tx ]

        } else{
            tileB[ty][tx] = 0f;
        }

        if((ph*TILE_WIDTH <)){

        }else{
            tileB[ty][tx] = 0f;
        }

        __syncthreads();


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