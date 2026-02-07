#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <iostream>

__global__ void matrix_vector_multiply_gpu_kernel(const float* A, const float* v, float* result, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += A[row * N + j] * v[j];
        }
        result[row] = sum;
    }
}

double time_matrix_vector_multiply_gpu(int N) {
    size_t matrix_size = N * N * sizeof(float);
    size_t vector_size = N * sizeof(float);

    std::vector<float> h_A(N * N);
    std::vector<float> h_v(N);
    std::vector<float> h_result(N);

    // Initialize matrix and vector
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < N; ++i) {
        h_v[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_v, *d_result;
    cudaMalloc(&d_A, matrix_size);
    cudaMalloc(&d_v, vector_size);
    cudaMalloc(&d_result, vector_size);

    cudaMemcpy(d_A, h_A.data(), matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), vector_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrix_vector_multiply_gpu_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_v, d_result, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_result.data(), d_result, vector_size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_v);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / 1000.0; // Convert to seconds
}