
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
