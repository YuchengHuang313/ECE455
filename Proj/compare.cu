#include <cuda_runtime.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "small_matmul.cuh"

#define MAT_SIZE 4

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
    std::cout << "========================================" << std::endl;

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
    initialize_random(h_A, total_elements);
    initialize_random(h_B, total_elements);
    initialize_random(h_C, total_elements);
    initialize_random(h_D, total_elements);

    // ========== SEPARATE LAYOUT (A,B,C,D) ==========
    std::cout << "\n[SEPARATE LAYOUT - A,B,C,D arrays]" << std::endl;

    // ========== CPU VERSION ==========
    auto cpu_start = std::chrono::high_resolution_clock::now();

    small_matmul_batched_cpu(h_A, h_B, h_C, h_D, h_out_cpu, num_matrices);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);

    // ========== CPU OMP VERSION ==========
    float* h_out_cpu_omp = new float[total_elements];
    int num_threads = omp_get_max_threads();
    auto cpu_omp_start = std::chrono::high_resolution_clock::now();

    small_matmul_batched_cpu_omp(h_A, h_B, h_C, h_D, h_out_cpu_omp, num_matrices);

    auto cpu_omp_end = std::chrono::high_resolution_clock::now();
    auto cpu_omp_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_omp_end - cpu_omp_start);

    // Free h_out_cpu_omp early since we'll use h_out_cpu for verification
    delete[] h_out_cpu_omp;

    // Save a subset of CPU results for verification, then free the full array
    const int verify_samples = std::min(10000, total_elements);  // Keep only 10k elements for verification
    float* h_verify_cpu = new float[verify_samples];
    std::memcpy(h_verify_cpu, h_out_cpu, verify_samples * sizeof(float));
    delete[] h_out_cpu;

    // ========== GPU VERSION ==========

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

    // Launch kernel
    const int threadsPerBlock = 64;
    int numBlocks = (num_matrices + threadsPerBlock - 1) / threadsPerBlock;

    // Check if we exceed max grid dimension on X axis (2^31-1 for modern GPUs)
    const long long maxGridDimX = 2147483647LL;  // 2^31 - 1
    if ((long long)numBlocks > maxGridDimX) {
        std::cerr << "Error: Number of blocks (" << numBlocks << ") exceeds maximum X grid dimension (" << maxGridDimX << ")" << std::endl;
        std::cerr << "This would require more than ~137 billion matrices. Consider batching your computation." << std::endl;

        // Free memory and exit
        delete[] h_verify_cpu;
        delete[] h_out_gpu;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_D);
        cudaFree(d_out);
        return 1;
    }

    auto kernel_start = std::chrono::high_resolution_clock::now();

    // Launch kernel directly
    dim3 blocks(numBlocks, 1);  // Use X dimension instead of Y
    dim3 threads(threadsPerBlock, 1);
    small_matmul_batched<<<blocks, threads>>>(d_A, d_B, d_C, d_D, d_out, num_matrices);

    CUDA_CHECK(cudaGetLastError());
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

    // ========== VERIFICATION ==========
    bool correct_gpu = compare_results(h_verify_cpu, h_out_gpu, verify_samples, 1e-4f);

    const int num_joints = 4;
    const int total_combined_input = num_matrices * num_joints * elements_per_matrix;
    const size_t combined_input_bytes = total_combined_input * sizeof(float);

    // Allocate combined format and CPU reference output
    float* h_matrices_combined = new float[total_combined_input];
    float* h_out_cpu_combined = new float[total_elements];
    float* h_out_cpu_omp_combined = new float[total_elements];
    float* h_out_gpu_combined = new float[total_elements];

    // Initialize with same random seed for reproducibility
    initialize_random(h_matrices_combined, total_combined_input);

    // Run CPU single-threaded version for combined format
    auto cpu_combined_start = std::chrono::high_resolution_clock::now();

    small_matmul_batched_combined_cpu(h_matrices_combined, h_out_cpu_combined, num_matrices, num_joints);

    auto cpu_combined_end = std::chrono::high_resolution_clock::now();
    auto cpu_combined_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_combined_end - cpu_combined_start);

    // Run CPU OpenMP version for combined format
    auto cpu_omp_combined_start = std::chrono::high_resolution_clock::now();

    small_matmul_batched_combined_cpu_omp(h_matrices_combined, h_out_cpu_omp_combined, num_matrices, num_joints);

    auto cpu_omp_combined_end = std::chrono::high_resolution_clock::now();
    auto cpu_omp_combined_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_omp_combined_end - cpu_omp_combined_start);

    // Allocate device memory for combined version
    float *d_matrices_combined, *d_out_combined;
    CUDA_CHECK(cudaMalloc(&d_matrices_combined, combined_input_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_combined, bytes));

    // Copy data to device
    auto gpu_combined_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_matrices_combined, h_matrices_combined, combined_input_bytes, cudaMemcpyHostToDevice));
    auto copy_combined_end = std::chrono::high_resolution_clock::now();

    // Launch combined kernel
    auto kernel_combined_start = std::chrono::high_resolution_clock::now();
    small_matmul_batched_combined<<<blocks, threads>>>(d_matrices_combined, d_out_combined, num_matrices, num_joints);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto kernel_combined_end = std::chrono::high_resolution_clock::now();

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_out_gpu_combined, d_out_combined, bytes, cudaMemcpyDeviceToHost));
    auto gpu_combined_end = std::chrono::high_resolution_clock::now();

    // Timing
    auto copy_combined_duration = std::chrono::duration_cast<std::chrono::microseconds>(copy_combined_end - gpu_combined_start);
    auto kernel_combined_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_combined_end - kernel_combined_start);
    auto gpu_combined_total = std::chrono::duration_cast<std::chrono::microseconds>(gpu_combined_end - gpu_combined_start);

    // Verify combined kernel correctness
    bool correct_combined = compare_results(h_out_cpu_combined, h_out_gpu_combined, total_elements, 1e-3f);
    bool correct_omp_combined = compare_results(h_out_cpu_combined, h_out_cpu_omp_combined, total_elements, 1e-5f);

    if (!correct_combined || !correct_omp_combined) {
        std::cerr << "ERROR: Combined layout verification failed!" << std::endl;
    }

    // Cleanup combined
    delete[] h_matrices_combined;
    delete[] h_out_cpu_combined;
    delete[] h_out_cpu_omp_combined;
    delete[] h_out_gpu_combined;
    CUDA_CHECK(cudaFree(d_matrices_combined));
    CUDA_CHECK(cudaFree(d_out_combined));

    // ========== PERFORMANCE SUMMARY ==========
    std::cout << "\n========================================" << std::endl;
    std::cout << "         PERFORMANCE SUMMARY" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n[EXECUTION TIME]" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "                          Separate    Combined" << std::endl;
    std::cout << "CPU (single-thread):      " << std::setw(7) << cpu_duration.count() / 1000.0 << " ms  " << std::setw(7)
              << cpu_combined_duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "CPU (OpenMP " << num_threads << " threads):    " << std::setw(7) << cpu_omp_duration.count() / 1000.0 << " ms  " << std::setw(7)
              << cpu_omp_combined_duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "GPU (kernel only):        " << std::setw(7) << kernel_duration.count() / 1000.0 << " ms  " << std::setw(7)
              << kernel_combined_duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "GPU (total w/ copy):      " << std::setw(7) << gpu_total_duration.count() / 1000.0 << " ms  " << std::setw(7)
              << gpu_combined_total.count() / 1000.0 << " ms" << std::endl;

    std::cout << "\n[SPEEDUP vs CPU Single-Threaded]" << std::endl;
    std::cout << std::setprecision(2);
    std::cout << "                          Separate    Combined" << std::endl;
    std::cout << "OpenMP (" << num_threads << " threads):      " << std::setw(7) << (double)cpu_duration.count() / cpu_omp_duration.count()
              << "x     " << std::setw(7) << (double)cpu_combined_duration.count() / cpu_omp_combined_duration.count() << "x" << std::endl;
    std::cout << "GPU (kernel only):        " << std::setw(7) << (double)cpu_duration.count() / kernel_duration.count() << "x     " << std::setw(7)
              << (double)cpu_combined_duration.count() / kernel_combined_duration.count() << "x" << std::endl;
    std::cout << "GPU (total):              " << std::setw(7) << (double)cpu_duration.count() / gpu_total_duration.count() << "x     " << std::setw(7)
              << (double)cpu_combined_duration.count() / gpu_combined_total.count() << "x" << std::endl;

    // Calculate GFLOPS
    // Each matrix multiply: 4×4×4 = 64 multiply-adds = 128 FLOPs
    // We do 3 multiplications per set: A×B, C×D, (A×B)×(C×D)
    long long total_flops = (long long)num_matrices * 3 * 128;
    double cpu_gflops = (double)total_flops / (cpu_duration.count() * 1e3);  // GFLOPS
    double cpu_omp_gflops = (double)total_flops / (cpu_omp_duration.count() * 1e3);
    double gpu_gflops = (double)total_flops / (kernel_duration.count() * 1e3);
    double cpu_combined_gflops = (double)total_flops / (cpu_combined_duration.count() * 1e3);
    double cpu_omp_combined_gflops = (double)total_flops / (cpu_omp_combined_duration.count() * 1e3);
    double gpu_combined_gflops = (double)total_flops / (kernel_combined_duration.count() * 1e3);

    std::cout << "\n[PERFORMANCE (GFLOPS)]" << std::endl;
    std::cout << std::setprecision(1);
    std::cout << "                          Separate    Combined" << std::endl;
    std::cout << "CPU (single):             " << std::setw(7) << cpu_gflops << "     " << std::setw(7) << cpu_combined_gflops << std::endl;
    std::cout << "CPU (OpenMP):             " << std::setw(7) << cpu_omp_gflops << "     " << std::setw(7) << cpu_omp_combined_gflops << std::endl;
    std::cout << "GPU (kernel):             " << std::setw(7) << gpu_gflops << "     " << std::setw(7) << gpu_combined_gflops << std::endl;

    std::cout << "\n[LAYOUT COMPARISON]" << std::endl;
    std::cout << std::setprecision(2);
    std::cout << "Combined vs Separate speedup:" << std::endl;
    std::cout << "  CPU single:        " << (double)cpu_duration.count() / cpu_combined_duration.count() << "x" << std::endl;
    std::cout << "  CPU OMP:           " << (double)cpu_omp_duration.count() / cpu_omp_combined_duration.count() << "x" << std::endl;
    std::cout << "  GPU (kernel):      " << (double)kernel_duration.count() / kernel_combined_duration.count() << "x" << std::endl;

    std::cout << "\n[VERIFICATION]" << std::endl;
    std::cout << "Separate layout:   " << (correct_gpu ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Combined layout:   " << (correct_combined && correct_omp_combined ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "========================================" << std::endl;

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
