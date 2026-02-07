#include <vector>
#include <chrono>
#include <iostream>

void matrix_vector_multiply_cpu(const std::vector<float>& A, const std::vector<float>& v, std::vector<float>& result, int N) {
    for (int i = 0; i < N; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += A[i * N + j] * v[j];
        }
        result[i] = sum;
    }
}

double time_matrix_vector_multiply_cpu(int N) {
    std::vector<float> A(N * N);
    std::vector<float> v(N);
    std::vector<float> result(N);

    // Initialize matrix and vector with some values
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < N; ++i) {
        v[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    auto start = std::chrono::high_resolution_clock::now();
    matrix_vector_multiply_cpu(A, v, result, N);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    return duration.count();
}