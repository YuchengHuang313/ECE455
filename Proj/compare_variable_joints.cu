#include <cuda_runtime.h>
#include <omp.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "small_matmul.cuh"

#define MAT_SIZE 4

// CSV output file path
const char* csv_filename = "compare_variable_joints_output.csv";

// Check CUDA errors
#define CUDA_CHECK(call)                                                                                                   \
    do {                                                                                                                   \
        cudaError_t err = call;                                                                                            \
        if (err != cudaSuccess) {                                                                                          \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                                                                            \
        }                                                                                                                  \
    } while (0)

bool has_unified_memory() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop.integrated == 1;
}

// ========== STRUCTURES FOR ORGANIZING DATA ==========

struct BenchmarkConfig {
    int num_ops;
    int num_joints;
    int threads_per_block;
    bool use_managed;
    std::string memory_type;
};

struct MemoryPointers {
    float* h_matrices;
    float* h_out_cpu;
    float* h_out_cpu_omp;
    float* h_out_gpu;
    float* d_matrices;
    float* d_out;
};

struct TimingResults {
    std::chrono::microseconds cpu;
    std::chrono::microseconds cpu_omp;
    std::chrono::microseconds h2d;
    std::chrono::microseconds kernel;
    std::chrono::microseconds d2h;
};

struct BenchmarkResults {
    double cpu_ms;
    double omp_ms;
    double gpu_kernel_ms;
    double transfer_ms;
    double total_ms;
    double cpu_gflops;
    double omp_gflops;
    double gpu_gflops;
    double speedup;
    bool correct;
};

// ========== CSV MANAGEMENT ==========

void write_csv_header() {
    std::ofstream file(csv_filename);
    if (file.is_open()) {
        file << "num_ops,num_joints,threads_per_block,memory_type,cpu_ms,omp_ms,gpu_kernel_ms,transfer_ms,total_ms,cpu_gflops,omp_gflops,gpu_gflops,"
                "speedup,"
                "correct\n";
        file.close();
    }
}

void append_csv_result(const BenchmarkConfig& config, const BenchmarkResults& results) {
    std::ofstream file(csv_filename, std::ios::app);
    if (file.is_open()) {
        file << config.num_ops << "," << config.num_joints << "," << config.threads_per_block << "," << config.memory_type << "," << std::fixed
             << std::setprecision(3) << results.cpu_ms << "," << results.omp_ms << "," << results.gpu_kernel_ms << "," << results.transfer_ms << ","
             << results.total_ms << "," << std::setprecision(2) << results.cpu_gflops << "," << results.omp_gflops << "," << results.gpu_gflops << ","
             << results.speedup << "," << (results.correct ? "1" : "0") << "\n";
        file.close();
    }
}

// ========== MEMORY MANAGEMENT ==========

MemoryPointers allocate_memory(int total_input_elements, int total_output_elements, bool use_managed) {
    MemoryPointers mem = {};

    const size_t input_bytes = total_input_elements * sizeof(float);
    const size_t output_bytes = total_output_elements * sizeof(float);

    if (use_managed) {
        // Use managed memory
        CUDA_CHECK(cudaMallocManaged(&mem.h_matrices, input_bytes));
        mem.h_out_cpu = new float[total_output_elements];
        mem.h_out_cpu_omp = new float[total_output_elements];
        CUDA_CHECK(cudaMallocManaged(&mem.h_out_gpu, output_bytes));
        // No separate device allocations needed for managed memory
        mem.d_matrices = nullptr;
        mem.d_out = nullptr;
    } else {
        // Use pinned memory for better GPU transfer performance
        CUDA_CHECK(cudaHostAlloc(&mem.h_matrices, input_bytes, cudaHostAllocDefault));
        mem.h_out_cpu = new float[total_output_elements];
        mem.h_out_cpu_omp = new float[total_output_elements];
        CUDA_CHECK(cudaHostAlloc(&mem.h_out_gpu, output_bytes, cudaHostAllocDefault));

        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&mem.d_matrices, input_bytes));
        CUDA_CHECK(cudaMalloc(&mem.d_out, output_bytes));
    }

    return mem;
}

void free_memory(MemoryPointers& mem, bool use_managed) {
    if (use_managed) {
        CUDA_CHECK(cudaFree(mem.h_matrices));
        delete[] mem.h_out_cpu;
        delete[] mem.h_out_cpu_omp;
        CUDA_CHECK(cudaFree(mem.h_out_gpu));
        // d_matrices and d_out are nullptr for managed memory
    } else {
        CUDA_CHECK(cudaFreeHost(mem.h_matrices));
        delete[] mem.h_out_cpu;
        delete[] mem.h_out_cpu_omp;
        CUDA_CHECK(cudaFreeHost(mem.h_out_gpu));
        CUDA_CHECK(cudaFree(mem.d_matrices));
        CUDA_CHECK(cudaFree(mem.d_out));
    }
}

// ========== CPU BENCHMARKING ==========

TimingResults run_cpu_benchmarks(MemoryPointers& mem, const BenchmarkConfig& config) {
    TimingResults timing = {};

    // Single-threaded CPU
    auto t_start = std::chrono::high_resolution_clock::now();
    small_matmul_batched_combined_cpu(mem.h_matrices, mem.h_out_cpu, config.num_ops, config.num_joints);
    auto t_end = std::chrono::high_resolution_clock::now();
    timing.cpu = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

    // OpenMP CPU
    t_start = std::chrono::high_resolution_clock::now();
    small_matmul_batched_combined_cpu_omp(mem.h_matrices, mem.h_out_cpu_omp, config.num_ops, config.num_joints);
    t_end = std::chrono::high_resolution_clock::now();
    timing.cpu_omp = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

    return timing;
}

// ========== GPU BENCHMARKING ==========

TimingResults run_gpu_benchmark(MemoryPointers& mem, const BenchmarkConfig& config, size_t input_bytes, size_t output_bytes) {
    TimingResults timing = {};

    // H2D transfer
    auto t_start = std::chrono::high_resolution_clock::now();
    if (!config.use_managed) {
        CUDA_CHECK(cudaMemcpy(mem.d_matrices, mem.h_matrices, input_bytes, cudaMemcpyHostToDevice));
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    timing.h2d = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

    // Kernel launch
    int numBlocks = (config.num_ops + config.threads_per_block - 1) / config.threads_per_block;
    dim3 blocks(numBlocks, 1);
    dim3 threads(config.threads_per_block, 1);

    t_start = std::chrono::high_resolution_clock::now();
    if (config.use_managed) {
        small_matmul_batched_combined<<<blocks, threads>>>(mem.h_matrices, mem.h_out_gpu, config.num_ops, config.num_joints);
    } else {
        small_matmul_batched_combined<<<blocks, threads>>>(mem.d_matrices, mem.d_out, config.num_ops, config.num_joints);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    t_end = std::chrono::high_resolution_clock::now();
    timing.kernel = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

    // D2H transfer
    t_start = std::chrono::high_resolution_clock::now();
    if (!config.use_managed) {
        CUDA_CHECK(cudaMemcpy(mem.h_out_gpu, mem.d_out, output_bytes, cudaMemcpyDeviceToHost));
    }
    t_end = std::chrono::high_resolution_clock::now();
    timing.d2h = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

    return timing;
}

// ========== RESULTS CALCULATION ==========

BenchmarkResults calculate_results(const BenchmarkConfig& config, const TimingResults& cpu_timing, const TimingResults& gpu_timing,
                                   MemoryPointers& mem, int total_output_elements) {
    BenchmarkResults results = {};

    // Convert timing to milliseconds
    results.cpu_ms = cpu_timing.cpu.count() / 1000.0;
    results.omp_ms = cpu_timing.cpu_omp.count() / 1000.0;
    results.gpu_kernel_ms = gpu_timing.kernel.count() / 1000.0;
    results.transfer_ms = (gpu_timing.h2d.count() + gpu_timing.d2h.count()) / 1000.0;
    results.total_ms = results.gpu_kernel_ms + results.transfer_ms;

    // Calculate GFLOPS
    // Each matrix multiply: 4×4×4 = 64 multiply-adds = 128 FLOPs
    // We do (num_joints - 1) multiplications per set (I × M0 × M1 × ... × Mn)
    long long total_flops = (long long)config.num_ops * (config.num_joints - 1) * 128;
    results.cpu_gflops = (double)total_flops / (cpu_timing.cpu.count() * 1e3);
    results.omp_gflops = (double)total_flops / (cpu_timing.cpu_omp.count() * 1e3);
    results.gpu_gflops = (double)total_flops / (gpu_timing.kernel.count() * 1e3);

    // Calculate speedup
    results.speedup = results.cpu_ms / results.total_ms;

    // Verify correctness (use adaptive tolerance for longer chains)
    float base_tolerance = 1e-3f;
    float n = (config.num_joints - 2);
    float gpu_tolerance = base_tolerance * (1.0f + 0.1f * n + 0.01f * n * n);
    bool correct_cpu_omp = compare_results(mem.h_out_cpu, mem.h_out_cpu_omp, total_output_elements, 1e-5f);
    bool correct_gpu = compare_results(mem.h_out_cpu, mem.h_out_gpu, total_output_elements, gpu_tolerance);
    results.correct = correct_cpu_omp && correct_gpu;

    return results;
}

// ========== RESULTS PRINTING ==========

void print_result_row(const BenchmarkConfig& config, const BenchmarkResults& results) {
    std::cout << std::setw(6) << std::right << config.num_joints << " | " << std::setw(8) << std::fixed << std::setprecision(3) << results.cpu_ms
              << " | " << std::setw(8) << results.omp_ms << " | " << std::setw(8) << results.gpu_kernel_ms << " | " << std::setw(9)
              << results.transfer_ms << " | " << std::setw(10) << results.total_ms << " | " << std::setw(6) << std::setprecision(1)
              << results.cpu_gflops << " | " << std::setw(6) << results.omp_gflops << " | " << std::setw(6) << results.gpu_gflops << " | "
              << std::setw(7) << std::setprecision(2) << results.speedup << "x | " << (results.correct ? "✓" : "✗") << std::endl;
}

void print_header(const BenchmarkConfig& config) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Matrix Chain Multiplication Test" << std::endl;
    std::cout << "  Number of matrix sets: " << config.num_ops << std::endl;
    std::cout << "  Threads per block: " << config.threads_per_block << std::endl;
    std::cout << "  OpenMP threads: " << omp_get_max_threads() << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Testing different numbers of joints (chain length)" << std::endl;
    std::cout << "Each set computes: I × M0 × M1 × ... × Mn" << std::endl;
    std::cout << std::endl;

    // Print table header
    std::cout << std::right;
    std::cout << std::setw(6) << "Joints" << " | " << std::setw(8) << "CPU (ms)" << " | " << std::setw(8) << "OMP (ms)" << " | " << std::setw(8)
              << "GPU (ms)" << " | " << std::setw(9) << "Xfer (ms)" << " | " << std::setw(9) << "Total (ms)" << " | " << std::setw(6) << "CPU GF"
              << " | " << std::setw(6) << "OMP GF" << " | " << std::setw(6) << "GPU GF" << " | " << std::setw(8) << "Speedup" << " | " << "OK"
              << std::endl;
    std::cout << "-------|----------|----------|----------|-----------|------------|--------|--------|--------|----------|----" << std::endl;
}

void print_legend() {
    std::cout << std::endl;
    std::cout << "Legend:" << std::endl;
    std::cout << "  Joints  = Number of 4x4 matrices in chain" << std::endl;
    std::cout << "  CPU/OMP = Single-threaded/OpenMP execution time" << std::endl;
    std::cout << "  GPU     = GPU kernel execution time (compute only)" << std::endl;
    std::cout << "  Xfer = Data transfer time (CPU→GPU + GPU→CPU copy time)" << std::endl;
    std::cout << "  Total = GPU + Xfer (realistic total GPU time including transfers)" << std::endl;
    std::cout << "  GF      = GFLOPS (billions of floating-point ops/sec)" << std::endl;
    std::cout << "  Speedup = CPU time / Total GPU time (realistic speedup)" << std::endl;
    std::cout << "  OK      = Verification passed (✓) or failed (✗)" << std::endl;
    std::cout << std::endl;
    std::cout << "Note: Speedup > 1.0 means GPU (including transfers) is faster than CPU" << std::endl;
    std::cout << "=========================================" << std::endl;
}

// ========== MAIN BENCHMARK FUNCTION ==========

void test_num_joints(const BenchmarkConfig& config) {
    const int elements_per_matrix = MAT_SIZE * MAT_SIZE;
    const int total_input_elements = config.num_ops * config.num_joints * elements_per_matrix;
    const int total_output_elements = config.num_ops * elements_per_matrix;
    const size_t input_bytes = total_input_elements * sizeof(float);
    const size_t output_bytes = total_output_elements * sizeof(float);

    // Allocate memory
    MemoryPointers mem = allocate_memory(total_input_elements, total_output_elements, config.use_managed);

    // Initialize data
    initialize_random(mem.h_matrices, total_input_elements);

    // Run CPU benchmarks
    TimingResults cpu_timing = run_cpu_benchmarks(mem, config);

    // Run GPU benchmark
    TimingResults gpu_timing = run_gpu_benchmark(mem, config, input_bytes, output_bytes);

    // Calculate and print results
    BenchmarkResults results = calculate_results(config, cpu_timing, gpu_timing, mem, total_output_elements);
    print_result_row(config, results);

    // Write to CSV
    append_csv_result(config, results);

    // Cleanup
    free_memory(mem, config.use_managed);
}

// ========== MAIN ==========

int main(int argc, char** argv) {
    // Show usage if help requested
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        std::cout << "Usage: " << argv[0] << " [num_matrices] [threads_per_block]" << std::endl;
        std::cout << "\nTests matrix chain multiplication with varying chain lengths (2-32 joints)" << std::endl;
        std::cout << "\nArguments:" << std::endl;
        std::cout << "  num_matrices       Number of matrix sets to process (default: 500000)" << std::endl;
        std::cout << "  threads_per_block  CUDA threads per block (default: 64)" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  " << argv[0] << " 1000000 128" << std::endl;
        return 0;
    }

    // Setup configuration
    BenchmarkConfig config;
    config.num_ops = (argc > 1) ? std::atoi(argv[1]) : 500000;
    config.threads_per_block = (argc > 2) ? std::atoi(argv[2]) : 64;
    bool has_unified = has_unified_memory();

    // Clear CSV file and write header
    std::remove(csv_filename);
    write_csv_header();

    // Test different numbers of joints
    int joint_configs[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32};
    int num_configs = sizeof(joint_configs) / sizeof(joint_configs[0]);

    // Determine which memory types to test
    std::vector<bool> memory_types_to_test;
    if (has_unified) {
        memory_types_to_test.push_back(true);   // Managed
        memory_types_to_test.push_back(false);  // Pinned
    } else {
        memory_types_to_test.push_back(false);  // Pinned only
    }

    for (bool use_managed : memory_types_to_test) {
        config.use_managed = use_managed;
        config.memory_type = use_managed ? "Managed" : "Pinned";

        // Print header for this memory type
        std::cout << "\n========================================" << std::endl;
        std::cout << "  Memory Type: " << config.memory_type << std::endl;
        std::cout << "========================================" << std::endl;
        print_header(config);

        for (int i = 0; i < num_configs; i++) {
            config.num_joints = joint_configs[i];
            test_num_joints(config);
        }
    }

    // Print legend
    print_legend();

    return 0;
}
