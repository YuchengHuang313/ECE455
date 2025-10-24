#include "small_matmul.cuh"

#include <cuda_runtime.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

#define MAT_SIZE 4

// CPU version of 4x4 matrix multiplication
void mul4x4_cpu(const float* A, const float* B, float* C) {
    for (int i = 0; i < MAT_SIZE; i++) {
        for (int j = 0; j < MAT_SIZE; j++) {
            float sum = 0.0f;
            for (int k = 0; k < MAT_SIZE; k++) {
                sum += A[i * MAT_SIZE + k] * B[k * MAT_SIZE + j];
            }
            C[i * MAT_SIZE + j] = sum;
        }
    }
}

// CPU version of batched matrix multiplication
void small_matmul_batched_cpu(const float* A, const float* B, const float* C, const float* D, float* out, int num_rows) {
    for (int row = 0; row < num_rows; row++) {
        float result_AB[MAT_SIZE * MAT_SIZE];
        float result_CD[MAT_SIZE * MAT_SIZE];

        // Compute A×B
        mul4x4_cpu(&A[row * MAT_SIZE * MAT_SIZE], &B[row * MAT_SIZE * MAT_SIZE], result_AB);

        // Compute C×D
        mul4x4_cpu(&C[row * MAT_SIZE * MAT_SIZE], &D[row * MAT_SIZE * MAT_SIZE], result_CD);

        // Compute (A×B)×(C×D)
        mul4x4_cpu(result_AB, result_CD, &out[row * MAT_SIZE * MAT_SIZE]);
    }
}

// OpenMP parallelized version of batched matrix multiplication
void small_matmul_batched_cpu_omp(const float* A, const float* B, const float* C, const float* D, float* out, int num_rows) {
#pragma omp parallel for schedule(static)
    for (int row = 0; row < num_rows; row++) {
        float result_AB[MAT_SIZE * MAT_SIZE];
        float result_CD[MAT_SIZE * MAT_SIZE];

        // Compute A×B
        mul4x4_cpu(&A[row * MAT_SIZE * MAT_SIZE], &B[row * MAT_SIZE * MAT_SIZE], result_AB);

        // Compute C×D
        mul4x4_cpu(&C[row * MAT_SIZE * MAT_SIZE], &D[row * MAT_SIZE * MAT_SIZE], result_CD);

        // Compute (A×B)×(C×D)
        mul4x4_cpu(result_AB, result_CD, &out[row * MAT_SIZE * MAT_SIZE]);
    }
}

// Helper function to initialize matrices with random values
void initialize_random(float* data, int size) {
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

// Helper function to compare two result arrays
bool compare_results(const float* cpu_result, const float* gpu_result, int size, float tolerance = 1e-4f) {
    for (int i = 0; i < size; i++) {
        float diff = std::abs(cpu_result[i] - gpu_result[i]);
        if (diff > tolerance) {
            std::cout << "Mismatch at index " << i << ": CPU=" << cpu_result[i] << ", GPU=" << gpu_result[i] << ", diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}

// Check CUDA errors
#define CUDA_CHECK(call)                                                                                                   \
    do {                                                                                                                   \
        cudaError_t err = call;                                                                                            \
        if (err != cudaSuccess) {                                                                                          \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                                                                            \
        }                                                                                                                  \
    } while (0)

int main(int argc, char** argv) {
    // Default number of matrices
    int num_matrices = 1000;
    if (argc > 1) {
        num_matrices = std::atoi(argv[1]);
    }

    std::cout << "Batched 4x4 Matrix Multiplication Test" << std::endl;
    std::cout << "Number of matrices: " << num_matrices << std::endl;
    std::cout << "Computing (A×B)×(C×D) for each set" << std::endl;
    std::cout << std::endl;

    // Calculate sizes
    const int elements_per_matrix = MAT_SIZE * MAT_SIZE;
    const int total_elements = num_matrices * elements_per_matrix;
    const size_t bytes = total_elements * sizeof(float);

    // Allocate host memory
    float* h_A = new float[total_elements];
    float* h_B = new float[total_elements];
    float* h_C = new float[total_elements];
    float* h_D = new float[total_elements];
    float* h_out_cpu = new float[total_elements];
    float* h_out_gpu = new float[total_elements];

    // Initialize with random data
    std::cout << "Initializing data..." << std::endl;
    initialize_random(h_A, total_elements);
    initialize_random(h_B, total_elements);
    initialize_random(h_C, total_elements);
    initialize_random(h_D, total_elements);

    // ========== CPU VERSION ==========
    std::cout << "\nRunning CPU (single-threaded) version..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();

    small_matmul_batched_cpu(h_A, h_B, h_C, h_D, h_out_cpu, num_matrices);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    std::cout << "CPU Time: " << cpu_duration.count() / 1000.0 << " ms" << std::endl;

    // ========== CPU OMP VERSION ==========
    float* h_out_cpu_omp = new float[total_elements];
    int num_threads = omp_get_max_threads();
    std::cout << "\nRunning CPU (OpenMP with " << num_threads << " threads) version..." << std::endl;
    auto cpu_omp_start = std::chrono::high_resolution_clock::now();

    small_matmul_batched_cpu_omp(h_A, h_B, h_C, h_D, h_out_cpu_omp, num_matrices);

    auto cpu_omp_end = std::chrono::high_resolution_clock::now();
    auto cpu_omp_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_omp_end - cpu_omp_start);
    std::cout << "CPU OMP Time: " << cpu_omp_duration.count() / 1000.0 << " ms" << std::endl;

    // Free h_out_cpu_omp early since we'll use h_out_cpu for verification
    delete[] h_out_cpu_omp;
    std::cout << "Freed h_out_cpu_omp (~" << bytes / (1024.0 * 1024.0) << " MB)" << std::endl;

    // Save a subset of CPU results for verification, then free the full array
    const int verify_samples = std::min(10000, total_elements);  // Keep only 10k elements for verification
    float* h_verify_cpu = new float[verify_samples];
    std::memcpy(h_verify_cpu, h_out_cpu, verify_samples * sizeof(float));
    delete[] h_out_cpu;
    std::cout << "Saved " << verify_samples << " elements for verification, freed h_out_cpu (~" << bytes / (1024.0 * 1024.0) << " MB)" << std::endl;

    // ========== GPU VERSION ==========
    std::cout << "\nRunning GPU version..." << std::endl;

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_D, *d_out;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMalloc(&d_D, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // Copy data to device
    auto gpu_start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D, h_D, bytes, cudaMemcpyHostToDevice));

    auto copy_to_device_end = std::chrono::high_resolution_clock::now();

    // Free input host arrays - no longer needed after copying to device
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_D;
    std::cout << "Freed input arrays h_A, h_B, h_C, h_D (~" << 4 * bytes / (1024.0 * 1024.0) << " MB)" << std::endl;

    // Launch kernel
    const int threadsPerBlock = 64;
    int numBlocks = (num_matrices + threadsPerBlock - 1) / threadsPerBlock;

    auto kernel_start = std::chrono::high_resolution_clock::now();

    small_matmul_batched_launch(d_A, d_B, d_C, d_D, d_out, num_matrices, numBlocks, threadsPerBlock);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto kernel_end = std::chrono::high_resolution_clock::now();

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost));

    auto gpu_end = std::chrono::high_resolution_clock::now();

    // Timing breakdown
    auto copy_to_duration = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_device_end - gpu_start);
    auto kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start);
    auto copy_back_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - kernel_end);
    auto gpu_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);

    std::cout << "GPU Total Time: " << gpu_total_duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "  - Copy to device: " << copy_to_duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "  - Kernel execution: " << kernel_duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "  - Copy back: " << copy_back_duration.count() / 1000.0 << " ms" << std::endl;

    // ========== VERIFICATION ==========
    std::cout << "\nVerifying results (using " << verify_samples << " sample elements)..." << std::endl;
    bool correct_gpu = compare_results(h_verify_cpu, h_out_gpu, verify_samples);

    if (correct_gpu) {
        std::cout << "✓ GPU results match CPU! Implementation is correct." << std::endl;
    } else {
        std::cout << "✗ GPU results DO NOT match!" << std::endl;
    }

    // ========== PERFORMANCE SUMMARY ==========
    std::cout << "\n========== PERFORMANCE SUMMARY ==========" << std::endl;
    std::cout << "CPU (single-thread):  " << cpu_duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "CPU (OpenMP " << num_threads << " threads): " << cpu_omp_duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "GPU (kernel only):    " << kernel_duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "GPU (total w/ copy):  " << gpu_total_duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << std::endl;

    std::cout << "Speedup vs single-threaded CPU:" << std::endl;
    std::cout << "  OpenMP:            " << (double)cpu_duration.count() / cpu_omp_duration.count() << "x" << std::endl;
    std::cout << "  GPU (kernel only): " << (double)cpu_duration.count() / kernel_duration.count() << "x" << std::endl;
    std::cout << "  GPU (total):       " << (double)cpu_duration.count() / gpu_total_duration.count() << "x" << std::endl;
    std::cout << std::endl;

    // Calculate GFLOPS
    // Each matrix multiply: 4×4×4 = 64 multiply-adds = 128 FLOPs
    // We do 3 multiplications per set: A×B, C×D, (A×B)×(C×D)
    long long total_flops = (long long)num_matrices * 3 * 128;
    double cpu_gflops = (double)total_flops / (cpu_duration.count() * 1e3);  // GFLOPS
    double cpu_omp_gflops = (double)total_flops / (cpu_omp_duration.count() * 1e3);
    double gpu_gflops = (double)total_flops / (kernel_duration.count() * 1e3);

    std::cout << "Performance (GFLOPS):" << std::endl;
    std::cout << "  CPU (single):  " << cpu_gflops << " GFLOPS" << std::endl;
    std::cout << "  CPU (OpenMP):  " << cpu_omp_gflops << " GFLOPS" << std::endl;
    std::cout << "  GPU (kernel):  " << gpu_gflops << " GFLOPS" << std::endl;

    // Cleanup (h_A, h_B, h_C, h_D, h_out_cpu_omp, h_out_cpu already freed earlier)
    delete[] h_verify_cpu;
    delete[] h_out_gpu;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
