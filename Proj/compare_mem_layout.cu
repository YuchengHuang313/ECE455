#include <cuda_runtime.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

#include "small_matmul.cuh"

#define MAT_SIZE 4

// Helper function for center-aligned text
std::string center(const std::string& str, int width) {
    int padding = width - str.length();
    int pad_left = padding / 2;
    int pad_right = padding - pad_left;
    return std::string(pad_left, ' ') + str + std::string(pad_right, ' ');
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
    // Show usage if help requested
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        std::cout << "Usage: " << argv[0] << " [num_matrices] [threads_per_block]" << std::endl;
        std::cout << "\nCompares separate (A,B,C,D arrays) vs combined (interleaved) memory layouts" << std::endl;
        std::cout << "\nArguments:" << std::endl;
        std::cout << "  num_matrices       Number of matrix sets to process (default: 1000000)" << std::endl;
        std::cout << "  threads_per_block  CUDA threads per block (default: 64)" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  " << argv[0] << " 1000000 128" << std::endl;
        return 0;
    }

    // Default parameters
    int num_matrices = 1000000;
    int threadsPerBlock = 64;

    if (argc > 1) {
        num_matrices = std::atoi(argv[1]);
    }
    if (argc > 2) {
        threadsPerBlock = std::atoi(argv[2]);
    }

    int num_threads = omp_get_max_threads();
    const int num_joints = 4;

    std::cout << "========================================" << std::endl;
    std::cout << "  Memory Layout Comparison Test" << std::endl;
    std::cout << "  Number of matrix sets: " << num_matrices << std::endl;
    std::cout << "  Number of joints (chain length): " << num_joints << std::endl;
    std::cout << "  Threads per block: " << threadsPerBlock << std::endl;
    std::cout << "  OpenMP threads: " << num_threads << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Calculate sizes
    const int elements_per_matrix = MAT_SIZE * MAT_SIZE;
    const int total_elements = num_matrices * elements_per_matrix;
    const size_t bytes = total_elements * sizeof(float);

    // Allocate host memory - use pinned memory for GPU transfers
    float *h_A, *h_B, *h_C, *h_D;
    CUDA_CHECK(cudaHostAlloc(&h_A, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_B, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_C, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_D, bytes, cudaHostAllocDefault));
    float* h_out_cpu = new float[total_elements];

    // Use pinned memory for GPU output to get better transfer performance
    float* h_out_gpu;
    CUDA_CHECK(cudaHostAlloc(&h_out_gpu, total_elements * sizeof(float), cudaHostAllocDefault));  // Initialize with random data
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

    auto copy_to_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_device_end - gpu_start).count();
    double copy_to_bandwidth = (bytes * 4 / (1024.0 * 1024.0 * 1024.0)) / (copy_to_duration_us / 1e6);
    std::cout << "Separate H->D: " << std::fixed << std::setprecision(2) << copy_to_duration_us / 1000.0 << " ms (" << copy_to_bandwidth << " GB/s)"
              << std::endl;

    // Free input host arrays - no longer needed after copying to device
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));
    CUDA_CHECK(cudaFreeHost(h_D));

    // Launch kernel
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

    auto copy_back_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - kernel_end).count();
    double copy_back_bandwidth = (bytes / (1024.0 * 1024.0 * 1024.0)) / (copy_back_duration_us / 1e6);
    std::cout << "Separate D->H: " << std::fixed << std::setprecision(2) << copy_back_duration_us / 1000.0 << " ms (" << copy_back_bandwidth
              << " GB/s)" << std::endl;
    std::cout << std::endl;

    // Timing breakdown
    auto copy_to_duration = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_device_end - gpu_start);
    auto kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start);
    auto copy_back_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - kernel_end);
    auto gpu_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);

    // ========== VERIFICATION ==========
    bool correct_gpu = compare_results(h_verify_cpu, h_out_gpu, verify_samples, 1e-4f);

    // Free verification data - no longer needed
    delete[] h_verify_cpu;
    CUDA_CHECK(cudaFreeHost(h_out_gpu));

    // Free GPU memory for separate layout
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaFree(d_out));

    // Ensure GPU is completely idle before next test
    CUDA_CHECK(cudaDeviceSynchronize());

    // Wait to let system settle (thermal, memory controller, PCIe state)
    std::cout << "Waiting 2 seconds before combined layout test..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // ========== COMBINED LAYOUT ==========

    const int total_combined_input = num_matrices * num_joints * elements_per_matrix;
    const size_t combined_input_bytes = total_combined_input * sizeof(float);

    // Allocate combined format and CPU reference output
    float* h_matrices_combined;
    CUDA_CHECK(cudaHostAlloc(&h_matrices_combined, combined_input_bytes, cudaHostAllocDefault));
    float* h_out_cpu_combined = new float[total_elements];
    float* h_out_cpu_omp_combined = new float[total_elements];

    // Use pinned memory for GPU output to get better transfer performance
    float* h_out_gpu_combined;
    CUDA_CHECK(cudaHostAlloc(&h_out_gpu_combined, total_elements * sizeof(float),
                             cudaHostAllocDefault));  // Initialize with same random seed for reproducibility
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

    auto copy_combined_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(copy_combined_end - gpu_combined_start).count();
    double copy_combined_to_bandwidth = (combined_input_bytes / (1024.0 * 1024.0 * 1024.0)) / (copy_combined_duration_us / 1e6);
    std::cout << "Combined H->D: " << std::fixed << std::setprecision(2) << copy_combined_duration_us / 1000.0 << " ms ("
              << copy_combined_to_bandwidth << " GB/s)" << std::endl;

    // Launch combined kernel
    auto kernel_combined_start = std::chrono::high_resolution_clock::now();
    small_matmul_batched_combined<<<blocks, threads>>>(d_matrices_combined, d_out_combined, num_matrices, num_joints);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto kernel_combined_end = std::chrono::high_resolution_clock::now();

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_out_gpu_combined, d_out_combined, bytes, cudaMemcpyDeviceToHost));
    auto gpu_combined_end = std::chrono::high_resolution_clock::now();

    auto copy_combined_back_us = std::chrono::duration_cast<std::chrono::microseconds>(gpu_combined_end - kernel_combined_end).count();
    double copy_combined_back_bandwidth = (bytes / (1024.0 * 1024.0 * 1024.0)) / (copy_combined_back_us / 1e6);
    std::cout << "Combined D->H: " << std::fixed << std::setprecision(2) << copy_combined_back_us / 1000.0 << " ms (" << copy_combined_back_bandwidth
              << " GB/s)" << std::endl;
    std::cout << std::endl;

    // Timing
    auto copy_combined_duration = std::chrono::duration_cast<std::chrono::microseconds>(copy_combined_end - gpu_combined_start);
    auto kernel_combined_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_combined_end - kernel_combined_start);
    auto gpu_combined_total = std::chrono::duration_cast<std::chrono::microseconds>(gpu_combined_end - gpu_combined_start);

    // Verify combined kernel correctness
    bool correct_combined = compare_results(h_out_cpu_combined, h_out_gpu_combined, total_elements, 1e-3f);
    bool correct_omp_combined = compare_results(h_out_cpu_combined, h_out_cpu_omp_combined, total_elements, 1e-5f);

    // Calculate GFLOPS
    // Each matrix multiply: 4×4×4 = 64 multiply-adds = 128 FLOPs
    // We do 3 multiplications per set: A×B, C×D, (A×B)×(C×D)
    long long total_flops = (long long)num_matrices * 3 * 128;
    double cpu_gflops = (double)total_flops / (cpu_duration.count() * 1e3);
    double cpu_omp_gflops = (double)total_flops / (cpu_omp_duration.count() * 1e3);
    double gpu_gflops = (double)total_flops / (kernel_duration.count() * 1e3);
    double cpu_combined_gflops = (double)total_flops / (cpu_combined_duration.count() * 1e3);
    double cpu_omp_combined_gflops = (double)total_flops / (cpu_omp_combined_duration.count() * 1e3);
    double gpu_combined_gflops = (double)total_flops / (kernel_combined_duration.count() * 1e3);

    // ========== PRINT RESULTS TABLE ==========
    std::cout << std::endl;
    std::cout << "Data transfer sizes:" << std::endl;
    std::cout << "  Separate layout input:  " << (bytes * 4) / (1024.0 * 1024.0) << " MB (4 arrays)" << std::endl;
    std::cout << "  Separate layout output: " << bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Combined layout input:  " << combined_input_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Combined layout output: " << bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << std::endl;
    std::cout << center("Layout", 10) << " | " << center("CPU (ms)", 10) << " | " << center("OMP (ms)", 10) << " | " << center("GPU (ms)", 10)
              << " | " << center("Xfer (ms)", 10) << " | " << center("Total (ms)", 11) << " | " << center("CPU GF", 8) << " | " << center("OMP GF", 8)
              << " | " << center("GPU GF", 8) << " | " << center("Speedup", 10) << " | " << center("OK", 4) << std::endl;
    std::cout << "-----------|------------|------------|------------|------------|-------------|----------|----------|----------|------------|------"
              << std::endl;

    // Separate layout row
    double separate_xfer_time = (copy_to_duration.count() + copy_back_duration.count()) / 1000.0;
    double separate_total_time = kernel_duration.count() / 1000.0 + separate_xfer_time;
    std::cout << std::setw(10) << std::right << "Separate" << " | " << std::setw(10) << std::fixed << std::setprecision(3)
              << cpu_duration.count() / 1000.0 << " | " << std::setw(10) << cpu_omp_duration.count() / 1000.0 << " | " << std::setw(10)
              << kernel_duration.count() / 1000.0 << " | " << std::setw(10) << separate_xfer_time << " | " << std::setw(11) << separate_total_time
              << " | " << std::setw(8) << std::setprecision(1) << cpu_gflops << " | " << std::setw(8) << cpu_omp_gflops << " | " << std::setw(8)
              << gpu_gflops << " | " << std::setw(9) << std::setprecision(2) << (cpu_duration.count() / 1000.0) / separate_total_time << "x | "
              << (correct_gpu ? "  ✓" : "  ✗") << std::endl;

    // Combined layout row
    auto copy_back_combined_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_combined_end - kernel_combined_end);
    double combined_xfer_time = (copy_combined_duration.count() + copy_back_combined_duration.count()) / 1000.0;
    double combined_total_time = kernel_combined_duration.count() / 1000.0 + combined_xfer_time;
    std::cout << std::setw(10) << std::right << "Combined" << " | " << std::setw(10) << std::fixed << std::setprecision(3)
              << cpu_combined_duration.count() / 1000.0 << " | " << std::setw(10) << cpu_omp_combined_duration.count() / 1000.0 << " | "
              << std::setw(10) << kernel_combined_duration.count() / 1000.0 << " | " << std::setw(10) << combined_xfer_time << " | " << std::setw(11)
              << combined_total_time << " | " << std::setw(8) << std::setprecision(1) << cpu_combined_gflops << " | " << std::setw(8)
              << cpu_omp_combined_gflops << " | " << std::setw(8) << gpu_combined_gflops << " | " << std::setw(9) << std::setprecision(2)
              << (cpu_combined_duration.count() / 1000.0) / combined_total_time << "x | "
              << (correct_combined && correct_omp_combined ? "  ✓" : "  ✗") << std::endl;

    std::cout << std::endl;
    std::cout << "Legend:" << std::endl;
    std::cout << "  Layout  = Memory layout (Separate: A,B,C,D arrays; Combined: interleaved)" << std::endl;
    std::cout << "  CPU/OMP = Single-threaded/OpenMP execution time" << std::endl;
    std::cout << "  GPU     = GPU kernel execution time (compute only)" << std::endl;
    std::cout << "  Xfer    = Data transfer time (CPU→GPU + GPU→CPU copy time)" << std::endl;
    std::cout << "  Total   = GPU + Xfer (realistic total GPU time including transfers)" << std::endl;
    std::cout << "  GF      = GFLOPS (billions of floating-point ops/sec)" << std::endl;
    std::cout << "  Speedup = CPU time / Total GPU time (realistic speedup)" << std::endl;
    std::cout << "  OK      = Verification passed (✓) or failed (✗)" << std::endl;

    std::cout << std::endl;
    std::cout << "Performance improvement (Combined vs Separate):" << std::endl;
    std::cout << "  CPU single:  " << std::setprecision(2) << (double)cpu_duration.count() / cpu_combined_duration.count() << "x" << std::endl;
    std::cout << "  CPU OMP:     " << (double)cpu_omp_duration.count() / cpu_omp_combined_duration.count() << "x" << std::endl;
    std::cout << "  GPU kernel:  " << (double)kernel_duration.count() / kernel_combined_duration.count() << "x" << std::endl;
    std::cout << "  GPU total:   " << separate_total_time / combined_total_time << "x" << std::endl;
    std::cout << std::endl;

    // Show the winner
    if (separate_total_time < combined_total_time) {
        std::cout << "Best GPU total time: " << std::setprecision(3) << separate_total_time << " ms (Separate layout)" << std::endl;
    } else {
        std::cout << "Best GPU total time: " << std::setprecision(3) << combined_total_time << " ms (Combined layout)" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    // Cleanup combined
    CUDA_CHECK(cudaFreeHost(h_matrices_combined));
    delete[] h_out_cpu_combined;
    delete[] h_out_cpu_omp_combined;
    CUDA_CHECK(cudaFreeHost(h_out_gpu_combined));
    CUDA_CHECK(cudaFree(d_matrices_combined));
    CUDA_CHECK(cudaFree(d_out_combined));

    return 0;
}
