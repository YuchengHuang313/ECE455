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
#include <vector>

#include "small_matmul.cuh"

#define MAT_SIZE 4

// ========== STRUCTURES FOR ORGANIZING DATA ==========

struct BenchmarkConfig {
    int num_matrices;
    int threads_per_block;
    int num_joints;
    bool use_unified;
};

struct MemoryPointers {
    float *h_A, *h_B, *h_C, *h_D;
    float* h_out_gpu;
    float *d_A, *d_B, *d_C, *d_D;
    float* d_out;
};

struct CombinedMemoryPointers {
    float* h_matrices_combined;
    float* h_out_gpu_combined;
    float* d_matrices_combined;
    float* d_out_combined;
};

struct TimingResults {
    std::chrono::microseconds cpu;
    std::chrono::microseconds cpu_omp;
    std::chrono::microseconds h2d;
    std::chrono::microseconds kernel;
    std::chrono::microseconds d2h;
};

// ========== HELPER FUNCTIONS ==========

std::string center(const std::string& str, int width) {
    if (str.length() >= static_cast<size_t>(width)) {
        return str;  // String is already wider than requested width
    }
    int padding = width - str.length();
    int pad_left = padding / 2;
    int pad_right = padding - pad_left;
    return std::string(pad_left, ' ') + str + std::string(pad_right, ' ');
}

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

// ========== MEMORY ALLOCATION ==========

MemoryPointers allocate_separate_memory(size_t bytes, bool use_unified) {
    MemoryPointers mem = {};

    if (use_unified) {
        CUDA_CHECK(cudaMallocManaged(&mem.h_A, bytes));
        CUDA_CHECK(cudaMallocManaged(&mem.h_B, bytes));
        CUDA_CHECK(cudaMallocManaged(&mem.h_C, bytes));
        CUDA_CHECK(cudaMallocManaged(&mem.h_D, bytes));
        CUDA_CHECK(cudaMallocManaged(&mem.h_out_gpu, bytes));
    } else {
        CUDA_CHECK(cudaHostAlloc(&mem.h_A, bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&mem.h_B, bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&mem.h_C, bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&mem.h_D, bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&mem.h_out_gpu, bytes, cudaHostAllocDefault));

        CUDA_CHECK(cudaMalloc(&mem.d_A, bytes));
        CUDA_CHECK(cudaMalloc(&mem.d_B, bytes));
        CUDA_CHECK(cudaMalloc(&mem.d_C, bytes));
        CUDA_CHECK(cudaMalloc(&mem.d_D, bytes));
        CUDA_CHECK(cudaMalloc(&mem.d_out, bytes));
    }

    return mem;
}

CombinedMemoryPointers allocate_combined_memory(size_t input_bytes, size_t output_bytes, bool use_unified) {
    CombinedMemoryPointers mem = {};

    if (use_unified) {
        CUDA_CHECK(cudaMallocManaged(&mem.h_matrices_combined, input_bytes));
        CUDA_CHECK(cudaMallocManaged(&mem.h_out_gpu_combined, output_bytes));
    } else {
        CUDA_CHECK(cudaHostAlloc(&mem.h_matrices_combined, input_bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&mem.h_out_gpu_combined, output_bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc(&mem.d_matrices_combined, input_bytes));
        CUDA_CHECK(cudaMalloc(&mem.d_out_combined, output_bytes));
    }

    return mem;
}

void free_separate_memory(MemoryPointers& mem, bool use_unified) {
    if (use_unified) {
        CUDA_CHECK(cudaFree(mem.h_A));
        CUDA_CHECK(cudaFree(mem.h_B));
        CUDA_CHECK(cudaFree(mem.h_C));
        CUDA_CHECK(cudaFree(mem.h_D));
        CUDA_CHECK(cudaFree(mem.h_out_gpu));
    } else {
        CUDA_CHECK(cudaFreeHost(mem.h_A));
        CUDA_CHECK(cudaFreeHost(mem.h_B));
        CUDA_CHECK(cudaFreeHost(mem.h_C));
        CUDA_CHECK(cudaFreeHost(mem.h_D));
        CUDA_CHECK(cudaFreeHost(mem.h_out_gpu));
        CUDA_CHECK(cudaFree(mem.d_A));
        CUDA_CHECK(cudaFree(mem.d_B));
        CUDA_CHECK(cudaFree(mem.d_C));
        CUDA_CHECK(cudaFree(mem.d_D));
        CUDA_CHECK(cudaFree(mem.d_out));
    }
}

void free_combined_memory(CombinedMemoryPointers& mem, bool use_unified) {
    if (use_unified) {
        CUDA_CHECK(cudaFree(mem.h_matrices_combined));
        CUDA_CHECK(cudaFree(mem.h_out_gpu_combined));
    } else {
        CUDA_CHECK(cudaFreeHost(mem.h_matrices_combined));
        CUDA_CHECK(cudaFreeHost(mem.h_out_gpu_combined));
        CUDA_CHECK(cudaFree(mem.d_matrices_combined));
        CUDA_CHECK(cudaFree(mem.d_out_combined));
    }
}

// ========== CPU BENCHMARKING ==========

TimingResults run_cpu_benchmarks_separate(MemoryPointers& mem, int num_matrices, int total_elements) {
    TimingResults timing = {};

    // Single-threaded CPU
    auto t_start = std::chrono::high_resolution_clock::now();
    small_matmul_batched_cpu(mem.h_A, mem.h_B, mem.h_C, mem.h_D, mem.h_out_gpu, num_matrices);
    auto t_end = std::chrono::high_resolution_clock::now();
    timing.cpu = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

    // OpenMP CPU
    float* h_out_cpu_omp = new float[total_elements];
    t_start = std::chrono::high_resolution_clock::now();
    small_matmul_batched_cpu_omp(mem.h_A, mem.h_B, mem.h_C, mem.h_D, h_out_cpu_omp, num_matrices);
    t_end = std::chrono::high_resolution_clock::now();
    timing.cpu_omp = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
    delete[] h_out_cpu_omp;

    return timing;
}

// ========== GPU BENCHMARKING (SEPARATE LAYOUT) ==========

TimingResults run_gpu_benchmark_separate(MemoryPointers& mem, const BenchmarkConfig& config, size_t bytes) {
    TimingResults timing = {};

    // H2D transfer
    auto t_start = std::chrono::high_resolution_clock::now();
    if (!config.use_unified) {
        CUDA_CHECK(cudaMemcpy(mem.d_A, mem.h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mem.d_B, mem.h_B, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mem.d_C, mem.h_C, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mem.d_D, mem.h_D, bytes, cudaMemcpyHostToDevice));
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    timing.h2d = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

    // Report transfer
    auto us_h2d = timing.h2d.count();
    double bw_h2d = (bytes * 4 / (1024.0 * 1024.0 * 1024.0)) / (us_h2d / 1e6);
    if (config.use_unified) {
        std::cout << "Separate H->D: N/A (managed memory - no transfer needed)" << std::endl;
    } else {
        std::cout << "Separate H->D: " << std::fixed << std::setprecision(2) << us_h2d / 1000.0 << " ms (" << bw_h2d << " GB/s)" << std::endl;
    }

    // Kernel launch
    int numBlocks = (config.num_matrices + config.threads_per_block - 1) / config.threads_per_block;
    dim3 blocks(numBlocks, 1);
    dim3 threads(config.threads_per_block, 1);

    t_start = std::chrono::high_resolution_clock::now();
    if (config.use_unified) {
        small_matmul_batched<<<blocks, threads>>>(mem.h_A, mem.h_B, mem.h_C, mem.h_D, mem.h_out_gpu, config.num_matrices);
    } else {
        small_matmul_batched<<<blocks, threads>>>(mem.d_A, mem.d_B, mem.d_C, mem.d_D, mem.d_out, config.num_matrices);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    t_end = std::chrono::high_resolution_clock::now();
    timing.kernel = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

    // D2H transfer
    t_start = std::chrono::high_resolution_clock::now();
    if (!config.use_unified) {
        CUDA_CHECK(cudaMemcpy(mem.h_out_gpu, mem.d_out, bytes, cudaMemcpyDeviceToHost));
    }
    t_end = std::chrono::high_resolution_clock::now();
    timing.d2h = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

    auto us_d2h = timing.d2h.count();
    double bw_d2h = (bytes / (1024.0 * 1024.0 * 1024.0)) / (us_d2h / 1e6);
    if (config.use_unified) {
        std::cout << "Separate D->H: N/A (managed memory - no transfer needed)" << std::endl;
    } else {
        std::cout << "Separate D->H: " << std::fixed << std::setprecision(2) << us_d2h / 1000.0 << " ms (" << bw_d2h << " GB/s)" << std::endl;
    }
    std::cout << std::endl;

    return timing;
}

// ========== GPU BENCHMARKING (COMBINED LAYOUT) ==========

TimingResults run_gpu_benchmark_combined(CombinedMemoryPointers& mem, const BenchmarkConfig& config, size_t input_bytes, size_t output_bytes) {
    TimingResults timing = {};

    // H2D transfer
    auto t_start = std::chrono::high_resolution_clock::now();
    if (!config.use_unified) {
        CUDA_CHECK(cudaMemcpy(mem.d_matrices_combined, mem.h_matrices_combined, input_bytes, cudaMemcpyHostToDevice));
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    timing.h2d = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

    auto us_h2d = timing.h2d.count();
    double bw_h2d = (input_bytes / (1024.0 * 1024.0 * 1024.0)) / (us_h2d / 1e6);
    if (config.use_unified) {
        std::cout << "Combined H->D: N/A (managed memory - no transfer needed)" << std::endl;
    } else {
        std::cout << "Combined H->D: " << std::fixed << std::setprecision(2) << us_h2d / 1000.0 << " ms (" << bw_h2d << " GB/s)" << std::endl;
    }

    // Kernel launch
    int numBlocks = (config.num_matrices + config.threads_per_block - 1) / config.threads_per_block;
    dim3 blocks(numBlocks, 1);
    dim3 threads(config.threads_per_block, 1);

    t_start = std::chrono::high_resolution_clock::now();
    if (config.use_unified) {
        small_matmul_batched_combined<<<blocks, threads>>>(mem.h_matrices_combined, mem.h_out_gpu_combined, config.num_matrices, config.num_joints);
    } else {
        small_matmul_batched_combined<<<blocks, threads>>>(mem.d_matrices_combined, mem.d_out_combined, config.num_matrices, config.num_joints);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    t_end = std::chrono::high_resolution_clock::now();
    timing.kernel = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

    // D2H transfer
    t_start = std::chrono::high_resolution_clock::now();
    if (!config.use_unified) {
        CUDA_CHECK(cudaMemcpy(mem.h_out_gpu_combined, mem.d_out_combined, output_bytes, cudaMemcpyDeviceToHost));
    }
    t_end = std::chrono::high_resolution_clock::now();
    timing.d2h = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

    auto us_d2h = timing.d2h.count();
    double bw_d2h = (output_bytes / (1024.0 * 1024.0 * 1024.0)) / (us_d2h / 1e6);
    if (config.use_unified) {
        std::cout << "Combined D->H: N/A (managed memory - no transfer needed)" << std::endl;
    } else {
        std::cout << "Combined D->H: " << std::fixed << std::setprecision(2) << us_d2h / 1000.0 << " ms (" << bw_d2h << " GB/s)" << std::endl;
    }
    std::cout << std::endl;

    return timing;
}

// ========== RESULTS PRINTING ==========

struct BenchmarkResult {
    std::string name;
    TimingResults timing;
    bool correct;
};

void print_results_table(const std::vector<BenchmarkResult>& results, int num_matrices, size_t bytes, size_t combined_input_bytes) {
    // Calculate GFLOPS
    long long total_flops = (long long)num_matrices * 3 * 128;

    std::cout << std::endl;
    std::cout << "Data transfer sizes:" << std::endl;
    std::cout << "  Separate layout input:  " << (bytes * 4) / (1024.0 * 1024.0) << " MB (4 arrays)" << std::endl;
    std::cout << "  Separate layout output: " << bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Combined layout input:  " << combined_input_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Combined layout output: " << bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << std::endl;

    std::cout << center("Layout", 18) << " | " << center("CPU (ms)", 10) << " | " << center("OMP (ms)", 10) << " | " << center("GPU (ms)", 10)
              << " | " << center("GPU Xfer (ms)", 13) << " | " << center("GPU Total (ms)", 14) << " | " << center("CPU GF", 8) << " | "
              << center("OMP GF", 8) << " | " << center("GPU GF", 8) << " | " << center("Speedup", 10) << " | " << center("OK", 4) << std::endl;
    std::cout << "-------------------|------------|------------|------------|---------------|----------------|----------|----------|----------|------"
                 "------|------"
              << std::endl;

    // Print each result row
    for (const auto& result : results) {
        double gflops_cpu = (double)total_flops / (result.timing.cpu.count() * 1e3);
        double gflops_cpu_omp = (double)total_flops / (result.timing.cpu_omp.count() * 1e3);
        double gflops_gpu = (double)total_flops / (result.timing.kernel.count() * 1e3);
        double ms_xfer = (result.timing.h2d.count() + result.timing.d2h.count()) / 1000.0;
        double ms_total = result.timing.kernel.count() / 1000.0 + ms_xfer;

        std::cout << std::setw(18) << std::right << result.name << " | " << std::setw(10) << std::fixed << std::setprecision(3)
                  << result.timing.cpu.count() / 1000.0 << " | " << std::setw(10) << result.timing.cpu_omp.count() / 1000.0 << " | " << std::setw(10)
                  << result.timing.kernel.count() / 1000.0 << " | " << std::setw(13) << ms_xfer << " | " << std::setw(14) << ms_total << " | "
                  << std::setw(8) << std::setprecision(1) << gflops_cpu << " | " << std::setw(8) << gflops_cpu_omp << " | " << std::setw(8)
                  << gflops_gpu << " | " << std::setw(9) << std::setprecision(2) << (result.timing.cpu.count() / 1000.0) / ms_total << "x | "
                  << (result.correct ? "  ✓" : "  ✗") << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Legend:" << std::endl;
    std::cout << "  Layout  = Memory layout and type (Managed/Pinned)" << std::endl;
    std::cout << "  CPU/OMP = Single-threaded/OpenMP execution time" << std::endl;
    std::cout << "  GPU     = GPU kernel execution time (compute only)" << std::endl;
    std::cout << "  Xfer    = Data transfer time (CPU→GPU + GPU→CPU copy time)" << std::endl;
    std::cout << "  Total   = GPU + Xfer (realistic total GPU time including transfers)" << std::endl;
    std::cout << "  GF      = GFLOPS (billions of floating-point ops/sec)" << std::endl;
    std::cout << "  Speedup = CPU time / Total GPU time (realistic speedup)" << std::endl;
    std::cout << "  OK      = Verification passed (✓) or failed (✗)" << std::endl;
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
}

void write_csv_results(const std::vector<BenchmarkResult>& results, int num_matrices, int threads_per_block) {
    const char* csv_filename = "compare_mem_layout_output.csv";
    std::remove(csv_filename);

    std::ofstream csv(csv_filename);
    if (!csv.is_open()) {
        std::cerr << "Warning: Could not open " << csv_filename << " for writing" << std::endl;
        return;
    }

    long long total_flops = (long long)num_matrices * 3 * 128;

    csv << "num_matrices,threads_per_block,layout,cpu_ms,omp_ms,gpu_ms,gpu_xfer_ms,gpu_total_ms,cpu_gflops,omp_gflops,gpu_gflops,speedup\n";
    csv << std::fixed << std::setprecision(3);

    for (const auto& result : results) {
        double gflops_cpu = (double)total_flops / (result.timing.cpu.count() * 1e3);
        double gflops_cpu_omp = (double)total_flops / (result.timing.cpu_omp.count() * 1e3);
        double gflops_gpu = (double)total_flops / (result.timing.kernel.count() * 1e3);
        double ms_xfer = (result.timing.h2d.count() + result.timing.d2h.count()) / 1000.0;
        double ms_total = result.timing.kernel.count() / 1000.0 + ms_xfer;

        csv << num_matrices << "," << threads_per_block << "," << result.name << "," << result.timing.cpu.count() / 1000.0 << ","
            << result.timing.cpu_omp.count() / 1000.0 << "," << result.timing.kernel.count() / 1000.0 << "," << ms_xfer << "," << ms_total << ","
            << std::setprecision(1) << gflops_cpu << "," << gflops_cpu_omp << "," << gflops_gpu << "," << std::setprecision(2)
            << (result.timing.cpu.count() / 1000.0) / ms_total << "\n";
    }

    csv.close();
    std::cout << "\nData written to: " << csv_filename << std::endl;
}

// ========== MAIN ==========

int main(int argc, char** argv) {
    // Show usage if help requested
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        std::cout << "Usage: " << argv[0] << " [num_matrices] [threads_per_block]" << std::endl;
        std::cout << "\nCompares separate (A,B,C,D arrays) vs combined (interleaved) memory layouts" << std::endl;
        std::cout << "Tests both managed and pinned memory types" << std::endl;
        std::cout << "\nArguments:" << std::endl;
        std::cout << "  num_matrices       Number of matrix sets to process (default: 1000000)" << std::endl;
        std::cout << "  threads_per_block  CUDA threads per block (default: 64)" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  " << argv[0] << " 1000000 128" << std::endl;
        return 0;
    }

    // Setup configuration
    BenchmarkConfig config;
    config.num_matrices = (argc > 1) ? std::atoi(argv[1]) : 1000000;
    config.threads_per_block = (argc > 2) ? std::atoi(argv[2]) : 64;
    config.num_joints = 4;
    bool has_unified = has_unified_memory();

    std::cout << "========================================" << std::endl;
    std::cout << "  Memory Layout Comparison Test" << std::endl;
    std::cout << "  Number of matrix sets: " << config.num_matrices << std::endl;
    std::cout << "  Number of joints (chain length): " << config.num_joints << std::endl;
    std::cout << "  Threads per block: " << config.threads_per_block << std::endl;
    std::cout << "  OpenMP threads: " << omp_get_max_threads() << std::endl;
    std::cout << "  System: " << (has_unified ? "Unified memory" : "Discrete GPU") << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Calculate sizes
    const int elements_per_matrix = MAT_SIZE * MAT_SIZE;
    const int total_elements = config.num_matrices * elements_per_matrix;
    const size_t bytes = total_elements * sizeof(float);
    const int total_combined_input = config.num_matrices * config.num_joints * elements_per_matrix;
    const size_t combined_input_bytes = total_combined_input * sizeof(float);

    std::vector<BenchmarkResult> results;

    // Helper to run a benchmark configuration
    auto run_benchmark = [&](const std::string& layout_name, bool use_managed, bool is_combined) {
        config.use_unified = use_managed;
        std::string full_name = layout_name + " (" + (use_managed ? "Managed" : "Pinned") + ")";

        std::cout << "\n[" << full_name << "]" << std::endl;

        TimingResults timing;
        bool correct;

        if (is_combined) {
            // Combined layout
            CombinedMemoryPointers comb_mem = allocate_combined_memory(combined_input_bytes, bytes, use_managed);
            initialize_random(comb_mem.h_matrices_combined, total_combined_input);

            // CPU benchmarks
            float* h_out_cpu_combined = new float[total_elements];
            auto t_start = std::chrono::high_resolution_clock::now();
            small_matmul_batched_combined_cpu(comb_mem.h_matrices_combined, h_out_cpu_combined, config.num_matrices, config.num_joints);
            auto t_end = std::chrono::high_resolution_clock::now();
            timing.cpu = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

            float* h_out_cpu_omp_combined = new float[total_elements];
            t_start = std::chrono::high_resolution_clock::now();
            small_matmul_batched_combined_cpu_omp(comb_mem.h_matrices_combined, h_out_cpu_omp_combined, config.num_matrices, config.num_joints);
            t_end = std::chrono::high_resolution_clock::now();
            timing.cpu_omp = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

            // GPU benchmark
            TimingResults gpu_timing = run_gpu_benchmark_combined(comb_mem, config, combined_input_bytes, bytes);
            timing.h2d = gpu_timing.h2d;
            timing.kernel = gpu_timing.kernel;
            timing.d2h = gpu_timing.d2h;

            correct = compare_results(h_out_cpu_combined, comb_mem.h_out_gpu_combined, total_elements, 1e-3f) &&
                      compare_results(h_out_cpu_combined, h_out_cpu_omp_combined, total_elements, 1e-5f);

            delete[] h_out_cpu_combined;
            delete[] h_out_cpu_omp_combined;
            free_combined_memory(comb_mem, use_managed);
        } else {
            // Separate layout
            MemoryPointers sep_mem = allocate_separate_memory(bytes, use_managed);
            initialize_random(sep_mem.h_A, total_elements);
            initialize_random(sep_mem.h_B, total_elements);
            initialize_random(sep_mem.h_C, total_elements);
            initialize_random(sep_mem.h_D, total_elements);

            timing = run_cpu_benchmarks_separate(sep_mem, config.num_matrices, total_elements);

            // Save CPU result for verification
            const int verify_samples = std::min(10000, total_elements);
            float* h_verify_cpu = new float[verify_samples];
            std::memcpy(h_verify_cpu, sep_mem.h_out_gpu, verify_samples * sizeof(float));

            TimingResults gpu_timing = run_gpu_benchmark_separate(sep_mem, config, bytes);
            timing.h2d = gpu_timing.h2d;
            timing.kernel = gpu_timing.kernel;
            timing.d2h = gpu_timing.d2h;

            correct = compare_results(h_verify_cpu, sep_mem.h_out_gpu, verify_samples, 1e-4f);
            delete[] h_verify_cpu;

            free_separate_memory(sep_mem, use_managed);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        results.push_back({full_name, timing, correct});

        // Wait between tests on discrete GPU
        if (!use_managed && results.size() < 4) {
            std::cout << "Waiting 1 second before next test..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    };

    // Run all benchmarks
    if (has_unified) {
        run_benchmark("Separate", true, false);   // Separate (Managed)
        run_benchmark("Separate", false, false);  // Separate (Pinned)
        run_benchmark("Combined", true, true);    // Combined (Managed)
        run_benchmark("Combined", false, true);   // Combined (Pinned)
    } else {
        // On discrete GPU, only run pinned versions
        run_benchmark("Separate", false, false);  // Separate (Pinned)
        run_benchmark("Combined", false, true);   // Combined (Pinned)
    }

    // Print consolidated results
    print_results_table(results, config.num_matrices, bytes, combined_input_bytes);
    write_csv_results(results, config.num_matrices, config.threads_per_block);

    return 0;
}
