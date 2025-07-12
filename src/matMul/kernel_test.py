import torch
import numpy as np
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import sys
import time

def load_cuda_module(cu_filename):
    """Load CUDA module from .cu file and return the extension object"""
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build full path to .cu file
    cu_filepath = os.path.join(current_dir, cu_filename)
    
    if not os.path.exists(cu_filepath):
        raise FileNotFoundError(f"CUDA file not found: {cu_filepath}")
    
    # Load the extension using the specified .cu file
    custom_matmul = load(
        name="custom_matmul",
        sources=[cu_filepath],
        verbose=True,
        with_cuda=True
    )
    
    return custom_matmul

# Default to matMul.cu if no argument provided
cu_filename = sys.argv[1] if len(sys.argv) > 1 else "matMul.cu"
custom_matmul = load_cuda_module(cu_filename)

# Register as a custom operator for torch.compile compatibility
@torch.library.custom_op("custom::matmul", mutates_args=())
def custom_matmul_op(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return custom_matmul.matmul_forward(A, B)

@custom_matmul_op.register_fake
def custom_matmul_fake(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.empty(A.shape[0], B.shape[1], dtype=A.dtype, device=A.device)

# Define a simple fully connected layer using our custom operator
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__() 
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        # Use our custom matrix multiplication
        output = custom_matmul_op(x, self.weight.t())
        return output + self.bias

def benchmark_matmul(A, B, iterations=100):
    """Benchmark custom and PyTorch matrix multiplication"""
    print(f"\nBenchmarking matrix multiplication ({A.shape} x {B.shape}) for {iterations} iterations...")
    
    # Warm up GPU
    for _ in range(10):
        _ = custom_matmul_op(A, B)
        _ = torch.mm(A, B)
    torch.cuda.synchronize()
    
    # Time custom implementation
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        result_custom = custom_matmul_op(A, B)
    torch.cuda.synchronize()
    custom_time = time.time() - start_time
    
    # Time PyTorch implementation
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        result_pytorch = torch.mm(A, B)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time
    
    # Calculate average times and speedup
    custom_avg = custom_time / iterations * 1000  # ms
    pytorch_avg = pytorch_time / iterations * 1000  # ms
    speedup = pytorch_time / custom_time
    
    print(f"Custom implementation: {custom_time:.4f}s total, {custom_avg:.4f}ms average")
    print(f"PyTorch implementation: {pytorch_time:.4f}s total, {pytorch_avg:.4f}ms average")
    print(f"Speedup: {speedup:.2f}x {'(custom faster)' if speedup > 1 else '(PyTorch faster)'}")
    
    # Verify correctness
    if torch.allclose(result_custom, result_pytorch, atol=1e-5):
        print("✓ Results match between implementations")
    else:
        print("✗ Results differ between implementations")
        print(f"Max difference: {torch.max(torch.abs(result_custom - result_pytorch))}")
    
    return custom_time, pytorch_time, speedup

# Test the custom operator
def test_custom_matmul():
    print("Testing custom matrix multiplication...")

    print(f"CUDA version {torch.version.cuda}")
    
    # Create test tensors
    A = torch.randn(32, 64, device='cuda', dtype=torch.float32).contiguous()
    B = torch.randn(64, 128, device='cuda', dtype=torch.float32).contiguous()

    # A = torch.from_numpy(np.array([[1,2,3], [4,5,6]], dtype=np.float32)).contiguous()
    # B = torch.from_numpy(np.array([[7,8],[9,10,], [11, 12]], dtype=np.float32)).contiguous()
    
    # Test our custom operation
    result_custom = custom_matmul_op(A, B)
    
    # Compare with PyTorch's built-in matmul
    result_pytorch = torch.mm(A, B)

    print(f"custom results: \n {result_custom}")
    print(f"Pytorch result: \n {result_pytorch}")
    
    # Check if results are close
    if torch.allclose(result_custom, result_pytorch, atol=1e-5):
        print(" Custom matmul matches PyTorch implementation")
    else:
        print(" Custom matmul differs from PyTorch implementation")
        print(f"Max difference: {torch.max(torch.abs(result_custom - result_pytorch))}")
    
    # Run timing benchmark
    benchmark_matmul(A, B, iterations=100)
    
    return result_custom

# Test with torch.compile    int col = blockIdx.x * blockDim.x + threadIdx.x; 
def test_with_compile():
    print("\nTesting with torch.compile...")
    print(f"")
    
    # Create a simple model
    model = CustomLinear(64, 32).cuda()
    
    # Compile the model
    compiled_model = torch.compile(model)
    
    # Test input
    x = torch.randn(16, 64, device='cuda', dtype=torch.float32).contiguous()
    
    # Test regular model
    with torch.no_grad():
        output_regular = model(x)
    
    # Test compiled model
    with torch.no_grad():
        output_compiled = compiled_model(x)
    
    # Check if outputs match
    if torch.allclose(output_regular, output_compiled, atol=1e-5):
        print("Compiled model matches regular model")
    else:
        print("Compiled model differs from regular model")
        print(f"Max difference: {torch.max(torch.abs(output_regular - output_compiled))}")
    
    print(" torch.compile completed successfully")

def run_comprehensive_benchmarks():
    """Run benchmarks with different matrix sizes"""
    print("\n" + "="*60)
    print("COMPREHENSIVE TIMING BENCHMARKS")
    print("="*60)
    
    # Test different matrix sizes
    test_sizes = [
        (32, 64, 128),   # Small
        (64, 128, 256),  # Medium
        (128, 256, 512), # Large
        (256, 512, 1024) # Very large
    ]
    
    results = []
    for m, k, n in test_sizes:
        print(f"\nTesting matrix size: ({m}, {k}) x ({k}, {n})")
        A = torch.randn(m, k, device='cuda', dtype=torch.float32).contiguous()
        B = torch.randn(k, n, device='cuda', dtype=torch.float32).contiguous()
        
        custom_time, pytorch_time, speedup = benchmark_matmul(A, B, iterations=100)
        results.append((m, k, n, custom_time, pytorch_time, speedup))
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Matrix Size':<20} {'Custom (s)':<12} {'PyTorch (s)':<13} {'Speedup':<10}")
    print("-" * 60)
    for m, k, n, custom_time, pytorch_time, speedup in results:
        size_str = f"({m},{k})x({k},{n})"
        print(f"{size_str:<20} {custom_time:<12.4f} {pytorch_time:<13.4f} {speedup:<10.2f}x")

if __name__ == "__main__":
    # Run tests
    test_custom_matmul()
    test_with_compile()
    
    # Run comprehensive benchmarks
    run_comprehensive_benchmarks()