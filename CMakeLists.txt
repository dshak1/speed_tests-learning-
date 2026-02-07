cmake_minimum_required(VERSION 3.18)
project(CudaPerformanceDemo LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Source files (CPU only for macOS)
set(CPU_SOURCES
    src/cpu/matrix_mult_cpu.cpp
    src/cpu/matrix_vector_cpu.cpp
)

# Create executables (CPU only)
add_executable(matrix_mult_demo src/main_matrix_mult.cpp ${CPU_SOURCES})
add_executable(matrix_vector_demo src/main_matrix_vector.cpp ${CPU_SOURCES})