
This repository demonstrates the performance differences between CPU and GPU implementations of algorithms with different time complexities, specifically O(n²) and O(n³) operations. since im on macbook i wont use nvidia gpu since its onot suported on macbook so instead what we can do is run it on cpu 


O(n³) - Matrix Multiplication
- PU Version: Traditional triple-nested loop implementation
- GPU Version: CUDA kernel with parallel computation

 O(n²) - Matrix-Vector Multiplication  
- CPU Version: Standard matrix-vector multiplication
- GPU Version: CUDA kernel implementation

 Prerequisites

- CMake (3.18 or later)
- C++ compiler with C++11 support (AppleClang works)
- For GPU comparison: NVIDIA GPU with CUDA Toolkit (not available on macOS/Apple Silicon)

 Building on macOS

Since CUDA is not supported on macOS, this builds CPU-only versions:

mkdir build
cd build
cmake ..
make

 Running

 Run matrix multiplication demo (O(n³))
./matrix_mult_demo

 Run matrix-vector multiplication demo (O(n²))
./matrix_vector_demo

 Expected Results

On CPU, you'll see how O(n³) operations (matrix multiplication) scale much worse than O(n²) operations (matrix-vector multiplication) as matrix size increases. GPU versions would show significant speedups, especially for O(n³) algorithms.

 For GPU Comparison

To see actual GPU vs CPU performance:
- Use Google Colab with GPU runtime
- Run on Linux/Windows with NVIDIA GPU
- Use cloud instances (AWS, GCP) with GPU support

 Examples to Try
   `
 Code Examples
1. Implement your own kernel: Try modifying the CUDA kernels to use different memory access patterns (coalesced vs. uncoalesced)
2. Add shared memory: Implement tile-based matrix multiplication using shared memory
3. Experiment with block sizes: Change blockDim and gridDim to see performance impact
4. Add more algorithms: Implement other O(n²) or O(n³) algorithms like:
   - Gaussian elimination
   - Convolution operations
   - N-body simulations 

 Topics to Read About
1. CUDA Memory Hierarchy: Global, shared, constant, and texture memory
2. Thread Synchronization: Using __syncthreads() and atomic operations
3. Memory Coalescing: Optimizing memory access patterns
4. Warp Divergence: Understanding how branch divergence affects performance
5. CUDA Streams: Asynchronous execution and overlapping compute with data transfer
6. Unified Memory: Automatic memory management between CPU and GPU
7. Profiling Tools: Using nvprof or Nsight for performance analysis

 Advanced Experiments
- Compare performance across different GPU architectures
- Implement CPU multithreading with OpenMP for fair comparison
- Add error checking and handling in CUDA code
- Experiment with different data types (float vs double)
- Try out CUDA libraries like cuBLAS for optimized implementations