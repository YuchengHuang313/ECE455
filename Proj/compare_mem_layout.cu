#include <cuda_runtime.h>
#include <omp.h>
#include <sys/stat.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
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
    auto t_cpu_start = std::chrono::high_resolution_clock::now();

    small_matmul_batched_cpu(h_A, h_B, h_C, h_D, h_out_cpu, num_matrices);

    auto t_cpu_end = std::chrono::high_resolution_clock::now();
    auto dur_cpu = std::chrono::duration_cast<std::chrono::microseconds>(t_cpu_end - t_cpu_start);

    // ========== CPU OMP VERSION ==========
    float* h_out_cpu_omp = new float[total_elements];
    auto t_cpu_omp_start = std::chrono::high_resolution_clock::now();

    small_matmul_batched_cpu_omp(h_A, h_B, h_C, h_D, h_out_cpu_omp, num_matrices);

    auto t_cpu_omp_end = std::chrono::high_resolution_clock::now();
    auto dur_cpu_omp = std::chrono::duration_cast<std::chrono::microseconds>(t_cpu_omp_end - t_cpu_omp_start);

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
    auto t_h2d_start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D, h_D, bytes, cudaMemcpyHostToDevice));

    auto t_h2d_end = std::chrono::high_resolution_clock::now();

    auto us_h2d = std::chrono::duration_cast<std::chrono::microseconds>(t_h2d_end - t_h2d_start).count();
    double bw_h2d = (bytes * 4 / (1024.0 * 1024.0 * 1024.0)) / (us_h2d / 1e6);
    std::cout << "Separate H->D: " << std::fixed << std::setprecision(2) << us_h2d / 1000.0 << " ms (" << bw_h2d << " GB/s)" << std::endl;

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

    auto t_kernel_start = std::chrono::high_resolution_clock::now();

    // Launch kernel directly
    dim3 blocks(numBlocks, 1);  // Use X dimension instead of Y
    dim3 threads(threadsPerBlock, 1);
    small_matmul_batched<<<blocks, threads>>>(d_A, d_B, d_C, d_D, d_out, num_matrices);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_kernel_end = std::chrono::high_resolution_clock::now();

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost));

    auto t_d2h_end = std::chrono::high_resolution_clock::now();

    auto us_d2h = std::chrono::duration_cast<std::chrono::microseconds>(t_d2h_end - t_kernel_end).count();
    double bw_d2h = (bytes / (1024.0 * 1024.0 * 1024.0)) / (us_d2h / 1e6);
    std::cout << "Separate D->H: " << std::fixed << std::setprecision(2) << us_d2h / 1000.0 << " ms (" << bw_d2h << " GB/s)" << std::endl;
    std::cout << std::endl;

    // Timing breakdown
    auto dur_h2d = std::chrono::duration_cast<std::chrono::microseconds>(t_h2d_end - t_h2d_start);
    auto dur_kernel = std::chrono::duration_cast<std::chrono::microseconds>(t_kernel_end - t_kernel_start);
    auto dur_d2h = std::chrono::duration_cast<std::chrono::microseconds>(t_d2h_end - t_kernel_end);
    auto dur_gpu_total = std::chrono::duration_cast<std::chrono::microseconds>(t_d2h_end - t_h2d_start);

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
    auto t_cpu_comb_start = std::chrono::high_resolution_clock::now();
    small_matmul_batched_combined_cpu(h_matrices_combined, h_out_cpu_combined, num_matrices, num_joints);
    auto t_cpu_comb_end = std::chrono::high_resolution_clock::now();
    auto dur_cpu_comb = std::chrono::duration_cast<std::chrono::microseconds>(t_cpu_comb_end - t_cpu_comb_start);

    // Run CPU OpenMP version for combined format
    auto t_cpu_omp_comb_start = std::chrono::high_resolution_clock::now();
    small_matmul_batched_combined_cpu_omp(h_matrices_combined, h_out_cpu_omp_combined, num_matrices, num_joints);
    auto t_cpu_omp_comb_end = std::chrono::high_resolution_clock::now();
    auto dur_cpu_omp_comb = std::chrono::duration_cast<std::chrono::microseconds>(t_cpu_omp_comb_end - t_cpu_omp_comb_start);

    // Allocate device memory for combined version
    float *d_matrices_combined, *d_out_combined;
    CUDA_CHECK(cudaMalloc(&d_matrices_combined, combined_input_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_combined, bytes));

    // Copy data to device
    auto t_h2d_comb_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_matrices_combined, h_matrices_combined, combined_input_bytes, cudaMemcpyHostToDevice));
    auto t_h2d_comb_end = std::chrono::high_resolution_clock::now();

    auto us_h2d_comb = std::chrono::duration_cast<std::chrono::microseconds>(t_h2d_comb_end - t_h2d_comb_start).count();
    double bw_h2d_comb = (combined_input_bytes / (1024.0 * 1024.0 * 1024.0)) / (us_h2d_comb / 1e6);
    std::cout << "Combined H->D: " << std::fixed << std::setprecision(2) << us_h2d_comb / 1000.0 << " ms (" << bw_h2d_comb << " GB/s)" << std::endl;

    // Launch combined kernel
    auto t_kernel_comb_start = std::chrono::high_resolution_clock::now();
    small_matmul_batched_combined<<<blocks, threads>>>(d_matrices_combined, d_out_combined, num_matrices, num_joints);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_kernel_comb_end = std::chrono::high_resolution_clock::now();

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_out_gpu_combined, d_out_combined, bytes, cudaMemcpyDeviceToHost));
    auto t_d2h_comb_end = std::chrono::high_resolution_clock::now();

    auto us_d2h_comb = std::chrono::duration_cast<std::chrono::microseconds>(t_d2h_comb_end - t_kernel_comb_end).count();
    double bw_d2h_comb = (bytes / (1024.0 * 1024.0 * 1024.0)) / (us_d2h_comb / 1e6);
    std::cout << "Combined D->H: " << std::fixed << std::setprecision(2) << us_d2h_comb / 1000.0 << " ms (" << bw_d2h_comb << " GB/s)" << std::endl;
    std::cout << std::endl;

    // Timing
    auto dur_h2d_comb = std::chrono::duration_cast<std::chrono::microseconds>(t_h2d_comb_end - t_h2d_comb_start);
    auto dur_kernel_comb = std::chrono::duration_cast<std::chrono::microseconds>(t_kernel_comb_end - t_kernel_comb_start);
    auto dur_gpu_total_comb = std::chrono::duration_cast<std::chrono::microseconds>(t_d2h_comb_end - t_h2d_comb_start);

    // Verify combined kernel correctness
    bool correct_combined = compare_results(h_out_cpu_combined, h_out_gpu_combined, total_elements, 1e-3f);
    bool correct_omp_combined = compare_results(h_out_cpu_combined, h_out_cpu_omp_combined, total_elements, 1e-5f);

    // Calculate GFLOPS
    // Each matrix multiply: 4×4×4 = 64 multiply-adds = 128 FLOPs
    // We do 3 multiplications per set: A×B, C×D, (A×B)×(C×D)
    long long total_flops = (long long)num_matrices * 3 * 128;
    double gflops_cpu = (double)total_flops / (dur_cpu.count() * 1e3);
    double gflops_cpu_omp = (double)total_flops / (dur_cpu_omp.count() * 1e3);
    double gflops_gpu = (double)total_flops / (dur_kernel.count() * 1e3);
    double gflops_cpu_comb = (double)total_flops / (dur_cpu_comb.count() * 1e3);
    double gflops_cpu_omp_comb = (double)total_flops / (dur_cpu_omp_comb.count() * 1e3);
    double gflops_gpu_comb = (double)total_flops / (dur_kernel_comb.count() * 1e3);

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
    double ms_sep_xfer = (dur_h2d.count() + dur_d2h.count()) / 1000.0;
    double ms_sep_total = dur_kernel.count() / 1000.0 + ms_sep_xfer;
    std::cout << std::setw(10) << std::right << "Separate" << " | " << std::setw(10) << std::fixed << std::setprecision(3) << dur_cpu.count() / 1000.0
              << " | " << std::setw(10) << dur_cpu_omp.count() / 1000.0 << " | " << std::setw(10) << dur_kernel.count() / 1000.0 << " | "
              << std::setw(10) << ms_sep_xfer << " | " << std::setw(11) << ms_sep_total << " | " << std::setw(8) << std::setprecision(1) << gflops_cpu
              << " | " << std::setw(8) << gflops_cpu_omp << " | " << std::setw(8) << gflops_gpu << " | " << std::setw(9) << std::setprecision(2)
              << (dur_cpu.count() / 1000.0) / ms_sep_total << "x | " << (correct_gpu ? "  ✓" : "  ✗") << std::endl;

    // Combined layout row
    auto dur_d2h_comb = std::chrono::duration_cast<std::chrono::microseconds>(t_d2h_comb_end - t_kernel_comb_end);
    double ms_comb_xfer = (dur_h2d_comb.count() + dur_d2h_comb.count()) / 1000.0;
    double ms_comb_total = dur_kernel_comb.count() / 1000.0 + ms_comb_xfer;
    std::cout << std::setw(10) << std::right << "Combined" << " | " << std::setw(10) << std::fixed << std::setprecision(3)
              << dur_cpu_comb.count() / 1000.0 << " | " << std::setw(10) << dur_cpu_omp_comb.count() / 1000.0 << " | " << std::setw(10)
              << dur_kernel_comb.count() / 1000.0 << " | " << std::setw(10) << ms_comb_xfer << " | " << std::setw(11) << ms_comb_total << " | "
              << std::setw(8) << std::setprecision(1) << gflops_cpu_comb << " | " << std::setw(8) << gflops_cpu_omp_comb << " | " << std::setw(8)
              << gflops_gpu_comb << " | " << std::setw(9) << std::setprecision(2) << (dur_cpu_comb.count() / 1000.0) / ms_comb_total << "x | "
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
    std::cout << "  CPU single:  " << std::setprecision(2) << (double)dur_cpu.count() / dur_cpu_comb.count() << "x" << std::endl;
    std::cout << "  CPU OMP:     " << (double)dur_cpu_omp.count() / dur_cpu_omp_comb.count() << "x" << std::endl;
    std::cout << "  GPU kernel:  " << (double)dur_kernel.count() / dur_kernel_comb.count() << "x" << std::endl;
    std::cout << "  GPU total:   " << ms_sep_total / ms_comb_total << "x" << std::endl;
    std::cout << std::endl;

    // Show the winner
    if (ms_sep_total < ms_comb_total) {
        std::cout << "Best GPU total time: " << std::setprecision(3) << ms_sep_total << " ms (Separate layout)" << std::endl;
    } else {
        std::cout << "Best GPU total time: " << std::setprecision(3) << ms_comb_total << " ms (Combined layout)" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    // ========== WRITE TO CSV FILE ==========
    const char* csv_filename = "compare_mem_layout_output.csv";

    // Delete old file and create fresh (overwrite mode)
    std::remove(csv_filename);

    std::ofstream csv(csv_filename);
    if (csv.is_open()) {
        // Always write header for fresh file
        csv << "num_matrices,threads_per_block,layout,cpu_ms,omp_ms,gpu_ms,xfer_ms,total_ms,cpu_gflops,omp_gflops,gpu_gflops,speedup\n";

        // Write data rows
        csv << std::fixed << std::setprecision(3);
        csv << num_matrices << "," << threadsPerBlock << ",Separate," << dur_cpu.count() / 1000.0 << "," << dur_cpu_omp.count() / 1000.0 << ","
            << dur_kernel.count() / 1000.0 << "," << ms_sep_xfer << "," << ms_sep_total << "," << std::setprecision(1) << gflops_cpu << ","
            << gflops_cpu_omp << "," << gflops_gpu << "," << std::setprecision(2) << (dur_cpu.count() / 1000.0) / ms_sep_total << "\n";

        csv << num_matrices << "," << threadsPerBlock << ",Combined," << dur_cpu_comb.count() / 1000.0 << "," << dur_cpu_omp_comb.count() / 1000.0
            << "," << dur_kernel_comb.count() / 1000.0 << "," << ms_comb_xfer << "," << ms_comb_total << "," << std::setprecision(1)
            << gflops_cpu_comb << "," << gflops_cpu_omp_comb << "," << gflops_gpu_comb << "," << std::setprecision(2)
            << (dur_cpu_comb.count() / 1000.0) / ms_comb_total << "\n";

        csv.close();
        std::cout << "\nData written to: " << csv_filename << std::endl;
    } else {
        std::cerr << "Warning: Could not open " << csv_filename << " for writing" << std::endl;
    }

    // Cleanup combined
    CUDA_CHECK(cudaFreeHost(h_matrices_combined));
    delete[] h_out_cpu_combined;
    delete[] h_out_cpu_omp_combined;
    CUDA_CHECK(cudaFreeHost(h_out_gpu_combined));
    CUDA_CHECK(cudaFree(d_matrices_combined));
    CUDA_CHECK(cudaFree(d_out_combined));

    return 0;
}
