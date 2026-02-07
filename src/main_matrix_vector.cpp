#include <iostream>
#include <iomanip>
#include <vector>

#include "cpu_timing.h"

int main() {
    std::cout << "Matrix-Vector Multiplication Performance Demo (CPU Only)" << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << "Note: GPU comparison requires CUDA-compatible hardware." << std::endl;
    std::cout << "For GPU vs CPU comparison, run on a machine with NVIDIA GPU." << std::endl << std::endl;

    std::vector<int> sizes = {1024, 2048, 4096, 8192};

    for (int N : sizes) {
        std::cout << "Matrix size: " << N << "x" << N << std::endl;

        double cpu_time = time_matrix_vector_multiply_cpu(N);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "CPU time: " << cpu_time << " seconds" << std::endl;
        std::cout << "Operations: " << (double)N * N << " multiplications" << std::endl;
        std::cout << "GFLOPS: " << ((double)N * N * 2) / (cpu_time * 1e9) << std::endl << std::endl;
    }

    return 0;
}