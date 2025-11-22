#include <cuda_runtime.h>
#include <omp.h>

#include <chrono>
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

void test_num_joints(int num_matrices, int num_joints, int threadsPerBlock) {
    const int elements_per_matrix = MAT_SIZE * MAT_SIZE;
    const int total_input_elements = num_matrices * num_joints * elements_per_matrix;
    const int total_output_elements = num_matrices * elements_per_matrix;
    const size_t input_bytes = total_input_elements * sizeof(float);
    const size_t output_bytes = total_output_elements * sizeof(float);

    // Allocate memory
    float* h_matrices = new float[total_input_elements];
    float* h_out_cpu = new float[total_output_elements];
    float* h_out_cpu_omp = new float[total_output_elements];
    float* h_out_gpu = new float[total_output_elements];

    // Initialize data
    initialize_random(h_matrices, total_input_elements);

    // ========== CPU SINGLE-THREADED ==========
    auto cpu_start = std::chrono::high_resolution_clock::now();
    small_matmul_batched_combined_cpu(h_matrices, h_out_cpu, num_matrices, num_joints);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);

    // ========== CPU OPENMP ==========
    auto cpu_omp_start = std::chrono::high_resolution_clock::now();
    small_matmul_batched_combined_cpu_omp(h_matrices, h_out_cpu_omp, num_matrices, num_joints);
    auto cpu_omp_end = std::chrono::high_resolution_clock::now();
    auto cpu_omp_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_omp_end - cpu_omp_start);

    // ========== GPU ==========
    float *d_matrices, *d_out;
    CUDA_CHECK(cudaMalloc(&d_matrices, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, output_bytes));

    auto gpu_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_matrices, h_matrices, input_bytes, cudaMemcpyHostToDevice));
    auto copy_end = std::chrono::high_resolution_clock::now();

    int numBlocks = (num_matrices + threadsPerBlock - 1) / threadsPerBlock;
    dim3 blocks(numBlocks, 1);
    dim3 threads(threadsPerBlock, 1);

    auto kernel_start = std::chrono::high_resolution_clock::now();
    small_matmul_batched_combined<<<blocks, threads>>>(d_matrices, d_out, num_matrices, num_joints);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto kernel_end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, output_bytes, cudaMemcpyDeviceToHost));
    auto gpu_end = std::chrono::high_resolution_clock::now();

    auto copy_duration = std::chrono::duration_cast<std::chrono::microseconds>(copy_end - gpu_start);
    auto kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start);
    auto gpu_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);

    // Verify correctness (use adaptive tolerance for longer chains)
    // Longer chains accumulate more floating-point error (grows quadratically)
    float base_tolerance = 1e-3f;
    float n = (num_joints - 2);
    float gpu_tolerance = base_tolerance * (1.0f + 0.1f * n + 0.01f * n * n);
    bool correct_cpu_omp = compare_results(h_out_cpu, h_out_cpu_omp, total_output_elements, 1e-5f);
    bool correct_gpu = compare_results(h_out_cpu, h_out_gpu, total_output_elements, gpu_tolerance);

    // Calculate GFLOPS
    // Each matrix multiply: 4×4×4 = 64 multiply-adds = 128 FLOPs
    // We do (num_joints - 1) multiplications per set (I × M0 × M1 × ... × Mn)
    long long total_flops = (long long)num_matrices * (num_joints - 1) * 128;
    double cpu_gflops = (double)total_flops / (cpu_duration.count() * 1e3);
    double cpu_omp_gflops = (double)total_flops / (cpu_omp_duration.count() * 1e3);
    double gpu_gflops = (double)total_flops / (kernel_duration.count() * 1e3);

    // Print results (right-align numbers for better table alignment)
    std::cout << std::setw(6) << std::right << num_joints << " | " << std::setw(8) << std::fixed << std::setprecision(3)
              << cpu_duration.count() / 1000.0 << " | " << std::setw(8) << cpu_omp_duration.count() / 1000.0 << " | " << std::setw(8)
              << kernel_duration.count() / 1000.0 << " | " << std::setw(6) << std::setprecision(1) << cpu_gflops << " | " << std::setw(6)
              << cpu_omp_gflops << " | " << std::setw(6) << gpu_gflops << " | " << std::setw(7) << std::setprecision(2)
              << (double)cpu_duration.count() / kernel_duration.count() << "x | " << (correct_cpu_omp && correct_gpu ? "✓" : "✗") << std::endl;

    // Cleanup
    delete[] h_matrices;
    delete[] h_out_cpu;
    delete[] h_out_cpu_omp;
    delete[] h_out_gpu;
    CUDA_CHECK(cudaFree(d_matrices));
    CUDA_CHECK(cudaFree(d_out));
}

int main(int argc, char** argv) {
    int num_matrices = 500000;
    int threadsPerBlock = 64;

    if (argc > 1) {
        num_matrices = std::atoi(argv[1]);
    }
    if (argc > 2) {
        threadsPerBlock = std::atoi(argv[2]);
    }

    int num_threads = omp_get_max_threads();

    std::cout << "========================================" << std::endl;
    std::cout << "  Matrix Chain Multiplication Test" << std::endl;
    std::cout << "  Number of matrix sets: " << num_matrices << std::endl;
    std::cout << "  Threads per block: " << threadsPerBlock << std::endl;
    std::cout << "  OpenMP threads: " << num_threads << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Testing different numbers of joints (chain length)" << std::endl;
    std::cout << "Each set computes: I × M0 × M1 × ... × Mn" << std::endl;
    std::cout << std::endl;

    // Print header with proper column alignment matching data rows
    std::cout << std::right;
    std::cout << std::setw(6) << "Joints" << " | " << std::setw(8) << "CPU (ms)" << " | " << std::setw(8) << "OMP (ms)" << " | " << std::setw(8)
              << "GPU (ms)" << " | " << std::setw(6) << "CPU GF" << " | " << std::setw(6) << "OMP GF" << " | " << std::setw(6) << "GPU GF" << " | "
              << std::setw(8) << "Speedup" << " | "
              << "OK" << std::endl;
    std::cout << "-------|----------|----------|----------|--------|--------|--------|----------|----" << std::endl;

    // Test different numbers of joints
    int joint_configs[] = {2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32};
    int num_configs = sizeof(joint_configs) / sizeof(joint_configs[0]);

    for (int i = 0; i < num_configs; i++) {
        test_num_joints(num_matrices, joint_configs[i], threadsPerBlock);
    }

    std::cout << std::endl;
    std::cout << "Legend:" << std::endl;
    std::cout << "  Joints  = Number of 4x4 matrices in chain" << std::endl;
    std::cout << "  CPU/OMP = Single-threaded/OpenMP execution time" << std::endl;
    std::cout << "  GPU     = GPU kernel execution time (excluding copy)" << std::endl;
    std::cout << "  GF      = GFLOPS (billions of floating-point ops/sec)" << std::endl;
    std::cout << "  Speedup = GPU speedup vs single-threaded CPU" << std::endl;
    std::cout << "  OK      = Verification passed (✓) or failed (✗)" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
