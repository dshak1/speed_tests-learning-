# CUDA Performance Demo

This repository demonstrates the performance differences between CPU and GPU implementations of algorithms with different time complexities, specifically O(n²) and O(n³) operations.

## Algorithms Implemented

### O(n³) - Matrix Multiplication
- **CPU Version**: Traditional triple-nested loop implementation
- **GPU Version**: CUDA kernel with parallel computation

### O(n²) - Matrix-Vector Multiplication  
- **CPU Version**: Standard matrix-vector multiplication
- **GPU Version**: CUDA kernel implementation

## Prerequisites

- CUDA Toolkit (11.0 or later)
- CMake (3.18 or later)
- C++ compiler with C++11 support
- NVIDIA GPU with CUDA support

to run it make a build folder cd into it and make all the files

mkdir build
cd build
cmake ..
make

## Running

```bash
# Run matrix multiplication comparison (O(n³))
./matrix_mult_demo

# Run matrix-vector multiplication comparison (O(n²))  
./matrix_vector_demo
```

## Expected Results

You should see significant speedup on the GPU versions, especially for larger matrix sizes. The O(n³) algorithm will show more dramatic improvements as n increases.

## Examples to Try

### Code Examples
1. **Implement your own kernel**: Try modifying the CUDA kernels to use different memory access patterns (coalesced vs. uncoalesced)
2. **Add shared memory**: Implement tile-based matrix multiplication using shared memory
3. **Experiment with block sizes**: Change blockDim and gridDim to see performance impact
4. **Add more algorithms**: Implement other O(n²) or O(n³) algorithms like:
   - Gaussian elimination
   - Convolution operations
   - N-body simulations

### Topics to Read About
1. **CUDA Memory Hierarchy**: Global, shared, constant, and texture memory
2. **Thread Synchronization**: Using __syncthreads() and atomic operations
3. **Memory Coalescing**: Optimizing memory access patterns
4. **Warp Divergence**: Understanding how branch divergence affects performance
5. **CUDA Streams**: Asynchronous execution and overlapping compute with data transfer
6. **Unified Memory**: Automatic memory management between CPU and GPU
7. **Profiling Tools**: Using nvprof or Nsight for performance analysis

### Advanced Experiments
- Compare performance across different GPU architectures
- Implement CPU multithreading with OpenMP for fair comparison
- Add error checking and handling in CUDA code
- Experiment with different data types (float vs double)
- Try out CUDA libraries like cuBLAS for optimized implementations