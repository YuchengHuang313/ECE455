#include <cuda_runtime.h>
#include <omp.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
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

struct GPUResults {
    double kernel_ms;
    double transfer_ms;
    double total_ms;
    double gflops;
    double power_w;
    bool valid;
};

struct BenchmarkResults {
    double cpu_ms;
    double omp_ms;
    double cpu_gflops;
    double omp_gflops;
    double speedup;
    bool correct;
    double cpu_power_w;
    double omp_power_w;
    GPUResults gpu_pinned;
    GPUResults gpu_managed;
};

// ========== CSV MANAGEMENT ==========

void write_csv_header(bool has_unified) {
    std::ofstream file(csv_filename);
    if (file.is_open()) {
        file << "num_ops,num_joints,threads_per_block,cpu_ms,omp_ms,"
                "gpu_pinned_kernel_ms,gpu_pinned_xfer_ms,gpu_pinned_total_ms,"
                "cpu_gflops,omp_gflops,gpu_pinned_gflops,speedup,correct,"
                "cpu_power_w,omp_power_w,gpu_pinned_power_w";
        if (has_unified) {
            file << ",gpu_managed_kernel_ms,gpu_managed_xfer_ms,gpu_managed_total_ms,"
                    "gpu_managed_gflops,gpu_managed_power_w";
        }
        file << "\n";
        file.close();
    }
}

void append_csv_result(const BenchmarkConfig& config, const BenchmarkResults& results) {
    std::ofstream file(csv_filename, std::ios::app);
    if (file.is_open()) {
        file << config.num_ops << "," << config.num_joints << "," << config.threads_per_block << "," << std::fixed << std::setprecision(6)
             << results.cpu_ms << "," << results.omp_ms << "," << results.gpu_pinned.kernel_ms << "," << results.gpu_pinned.transfer_ms << ","
             << results.gpu_pinned.total_ms << "," << results.cpu_gflops << "," << results.omp_gflops << "," << results.gpu_pinned.gflops << ","
             << results.speedup << "," << (results.correct ? "1" : "0") << "," << results.cpu_power_w << "," << results.omp_power_w << ","
             << results.gpu_pinned.power_w;
        if (results.gpu_managed.valid) {
            file << "," << std::setprecision(6) << results.gpu_managed.kernel_ms << "," << results.gpu_managed.transfer_ms << ","
                 << results.gpu_managed.total_ms << "," << results.gpu_managed.gflops << "," << results.gpu_managed.power_w;
        }
        file << "\n";
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
    } else {
        CUDA_CHECK(cudaFreeHost(mem.h_matrices));
        delete[] mem.h_out_cpu;
        delete[] mem.h_out_cpu_omp;
        CUDA_CHECK(cudaFreeHost(mem.h_out_gpu));
        CUDA_CHECK(cudaFree(mem.d_matrices));
        CUDA_CHECK(cudaFree(mem.d_out));
    }
}

// ========== POWER MEASUREMENT ==========

// Read Jetson power from sysfs
double read_jetson_power() {
    // VDD_IN rail (channel 1) measures total system input power
    const char* voltage_path = "/sys/devices/platform/bus@0/c240000.i2c/i2c-1/1-0040/hwmon/hwmon1/in1_input";
    const char* current_path = "/sys/devices/platform/bus@0/c240000.i2c/i2c-1/1-0040/hwmon/hwmon1/curr1_input";

    std::ifstream volt_file(voltage_path);
    std::ifstream curr_file(current_path);

    if (volt_file.is_open() && curr_file.is_open()) {
        int voltage_mv, current_ma;
        volt_file >> voltage_mv;
        curr_file >> current_ma;
        volt_file.close();
        curr_file.close();

        // Calculate power in watts: (mV * mA) / 1,000,000
        return (voltage_mv * current_ma) / 1000000.0;
    }

    return 0.0;
}

// Power sampler class for continuous measurement
class PowerSampler {
   private:
    std::atomic<bool> running;
    std::thread sampler_thread;
    std::vector<double> samples;

    void sample_loop() {
        while (running) {
            double power = read_jetson_power();
            if (power > 0.0) {
                samples.push_back(power);
            }
        }
    }

   public:
    PowerSampler() : running(false) {}

    void start() {
        samples.clear();
        running = true;
        sampler_thread = std::thread(&PowerSampler::sample_loop, this);
    }

    double stop() {
        running = false;
        if (sampler_thread.joinable()) {
            sampler_thread.join();
        }

        if (samples.empty()) {
            return 0.0;
        }

        double sum = 0.0;
        for (double s : samples) {
            sum += s;
        }
        return sum / samples.size();
    }
};

// ========== BENCHMARKING FUNCTIONS ==========

// Run CPU benchmark only
double run_cpu_benchmark(MemoryPointers& mem, const BenchmarkConfig& config) {
    auto t_start = std::chrono::high_resolution_clock::now();
    small_matmul_batched_combined_cpu(mem.h_matrices, mem.h_out_cpu, config.num_ops, config.num_joints);
    auto t_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;  // Return ms
}

// Run CPU+OMP benchmark only
double run_omp_benchmark(MemoryPointers& mem, const BenchmarkConfig& config) {
    auto t_start = std::chrono::high_resolution_clock::now();
    small_matmul_batched_combined_cpu_omp(mem.h_matrices, mem.h_out_cpu_omp, config.num_ops, config.num_joints);
    auto t_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;  // Return ms
}

// Run GPU benchmark (including transfers)
struct GPUTiming {
    double kernel_ms;
    double transfer_ms;
    double total_ms;
};

GPUTiming run_gpu_benchmark(MemoryPointers& mem, const BenchmarkConfig& config, size_t input_bytes, size_t output_bytes) {
    GPUTiming timing = {};

    // Use CUDA events for more accurate GPU kernel timing
    cudaEvent_t start_h2d, stop_h2d, start_kernel, stop_kernel, start_d2h, stop_d2h;
    CUDA_CHECK(cudaEventCreate(&start_h2d));
    CUDA_CHECK(cudaEventCreate(&stop_h2d));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));
    CUDA_CHECK(cudaEventCreate(&start_d2h));
    CUDA_CHECK(cudaEventCreate(&stop_d2h));

    // H2D transfer
    CUDA_CHECK(cudaEventRecord(start_h2d));
    if (!config.use_managed) {
        CUDA_CHECK(cudaMemcpy(mem.d_matrices, mem.h_matrices, input_bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop_h2d));
    CUDA_CHECK(cudaEventSynchronize(stop_h2d));

    // Kernel launch
    int numBlocks = (config.num_ops + config.threads_per_block - 1) / config.threads_per_block;
    dim3 blocks(numBlocks, 1);
    dim3 threads(config.threads_per_block, 1);

    CUDA_CHECK(cudaEventRecord(start_kernel));
    if (config.use_managed) {
        small_matmul_batched_combined<<<blocks, threads>>>(mem.h_matrices, mem.h_out_gpu, config.num_ops, config.num_joints);
    } else {
        small_matmul_batched_combined<<<blocks, threads>>>(mem.d_matrices, mem.d_out, config.num_ops, config.num_joints);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));  // D2H transfer
    CUDA_CHECK(cudaEventRecord(start_d2h));
    if (!config.use_managed) {
        CUDA_CHECK(cudaMemcpy(mem.h_out_gpu, mem.d_out, output_bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop_d2h));
    CUDA_CHECK(cudaEventSynchronize(stop_d2h));

    // Calculate elapsed times
    float h2d_ms, kernel_ms, d2h_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start_h2d, stop_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start_d2h, stop_d2h));

    timing.kernel_ms = kernel_ms;
    timing.transfer_ms = h2d_ms + d2h_ms;
    timing.total_ms = timing.kernel_ms + timing.transfer_ms;

    // Cleanup events
    CUDA_CHECK(cudaEventDestroy(start_h2d));
    CUDA_CHECK(cudaEventDestroy(stop_h2d));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));
    CUDA_CHECK(cudaEventDestroy(start_d2h));
    CUDA_CHECK(cudaEventDestroy(stop_d2h));

    return timing;
}

// ========== RESULTS PRINTING ==========

void print_header(const BenchmarkConfig& config, bool has_unified) {
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
    std::cout << std::setw(6) << "Joints" << " | " << std::setw(8) << "CPU (ms)" << " | " << std::setw(8) << "OMP (ms)" << " | " << std::setw(11)
              << "GPU_P (ms)" << " | " << std::setw(12) << "GPU_P_X (ms)" << " | " << std::setw(12) << "GPU_P_T (ms)" << " | ";
    if (has_unified) {
        std::cout << std::setw(11) << "GPU_M (ms)" << " | " << std::setw(12) << "GPU_M_X (ms)" << " | " << std::setw(12) << "GPU_M_T (ms)" << " | ";
    }
    std::cout << std::setw(6) << "CPU GF" << " | " << std::setw(6) << "OMP GF" << " | " << std::setw(8) << "GPU_P_GF" << " | ";
    if (has_unified) {
        std::cout << std::setw(8) << "GPU_M_GF" << " | ";
    }
    std::cout << std::setw(8) << "Speedup" << " | " << std::setw(7) << "CPU_W" << " | " << std::setw(7) << "OMP_W" << " | " << std::setw(7)
              << "GPU_P_W" << " | ";
    if (has_unified) {
        std::cout << std::setw(7) << "GPU_M_W" << " | ";
    }
    std::cout << "OK" << std::endl;

    // Print separator
    std::cout << "-------|-";
    std::cout << std::string(8, '-') << "-|-";   // CPU (ms)
    std::cout << std::string(8, '-') << "-|-";   // OMP (ms)
    std::cout << std::string(11, '-') << "-|-";  // GPU_P (ms)
    std::cout << std::string(12, '-') << "-|-";  // GPU_P_X (ms)
    std::cout << std::string(12, '-') << "-|-";  // GPU_P_T (ms)
    if (has_unified) {
        std::cout << std::string(11, '-') << "-|-";  // GPU_M (ms)
        std::cout << std::string(12, '-') << "-|-";  // GPU_M_X (ms)
        std::cout << std::string(12, '-') << "-|-";  // GPU_M_T (ms)
    }
    std::cout << std::string(6, '-') << "-|-";  // CPU GF
    std::cout << std::string(6, '-') << "-|-";  // OMP GF
    std::cout << std::string(8, '-') << "-|-";  // GPU_P_GF
    if (has_unified) {
        std::cout << std::string(8, '-') << "-|-";  // GPU_M_GF
    }
    std::cout << std::string(8, '-') << "-|-";  // Speedup
    std::cout << std::string(7, '-') << "-|-";  // CPU_W
    std::cout << std::string(7, '-') << "-|-";  // OMP_W
    std::cout << std::string(7, '-') << "-|-";  // GPU_P_W
    if (has_unified) {
        std::cout << std::string(7, '-') << "-|-";  // GPU_M_W
    }
    std::cout << "--" << std::endl;  // OK
}

void print_result_row(const BenchmarkConfig& config, const BenchmarkResults& results, bool has_unified) {
    std::cout << std::setw(6) << std::right << config.num_joints << " | " << std::setw(8) << std::fixed << std::setprecision(3) << results.cpu_ms
              << " | " << std::setw(8) << results.omp_ms << " | " << std::setw(11) << results.gpu_pinned.kernel_ms << " | " << std::setw(12)
              << results.gpu_pinned.transfer_ms << " | " << std::setw(12) << results.gpu_pinned.total_ms << " | ";
    if (has_unified) {
        std::cout << std::setw(11) << results.gpu_managed.kernel_ms << " | " << std::setw(12) << results.gpu_managed.transfer_ms << " | "
                  << std::setw(12) << results.gpu_managed.total_ms << " | ";
    }
    std::cout << std::setw(6) << std::setprecision(1) << results.cpu_gflops << " | " << std::setw(6) << results.omp_gflops << " | " << std::setw(8)
              << results.gpu_pinned.gflops << " | ";
    if (has_unified) {
        std::cout << std::setw(8) << results.gpu_managed.gflops << " | ";
    }
    std::cout << std::setw(7) << std::setprecision(2) << results.speedup << "x | " << std::setw(5) << std::setprecision(4) << results.cpu_power_w
              << "W | " << std::setw(5) << results.omp_power_w << "W | " << std::setw(6) << results.gpu_pinned.power_w << "W | ";
    if (has_unified) {
        std::cout << std::setw(6) << results.gpu_managed.power_w << "W | ";
    }
    std::cout << (results.correct ? "✓" : "✗") << std::endl;
}

void print_legend() {
    std::cout << std::endl;
    std::cout << "Legend:" << std::endl;
    std::cout << "  Joints    = Number of 4x4 matrices in chain" << std::endl;
    std::cout << "  CPU/OMP   = Single-threaded/OpenMP execution time" << std::endl;
    std::cout << "  GPU_P     = GPU Pinned memory kernel time (compute only)" << std::endl;
    std::cout << "  GPU_P_X   = GPU Pinned transfer time (CPU↔GPU)" << std::endl;
    std::cout << "  GPU_P_T   = GPU Pinned total time (kernel + transfer)" << std::endl;
    std::cout << "  GPU_M     = GPU Managed memory kernel time (compute only)" << std::endl;
    std::cout << "  GPU_M_X   = GPU Managed transfer time (minimal/zero)" << std::endl;
    std::cout << "  GPU_M_T   = GPU Managed total time (kernel + transfer)" << std::endl;
    std::cout << "  GF        = GFLOPS (billions of floating-point ops/sec)" << std::endl;
    std::cout << "  Speedup   = CPU time / GPU_P_T time" << std::endl;
    std::cout << "  *_W       = Average power consumption in watts" << std::endl;
    std::cout << "  OK        = Verification passed (✓) or failed (✗)" << std::endl;
    std::cout << std::endl;
    std::cout << "Note: GPU_M columns only appear on unified memory devices" << std::endl;
    std::cout << "========================================" << std::endl;
}

// ========== MAIN BENCHMARK FUNCTION ==========

void test_num_joints(const BenchmarkConfig& config, bool has_unified) {
    const int elements_per_matrix = MAT_SIZE * MAT_SIZE;
    const int total_input_elements = config.num_ops * config.num_joints * elements_per_matrix;
    const int total_output_elements = config.num_ops * elements_per_matrix;
    const size_t input_bytes = total_input_elements * sizeof(float);
    const size_t output_bytes = total_output_elements * sizeof(float);

    BenchmarkResults results = {};

    // Initialize results
    results.gpu_pinned.valid = true;
    results.gpu_managed.valid = false;

    // ===== Allocate memory for CPU/OMP (use pinned) =====
    MemoryPointers cpu_mem = allocate_memory(total_input_elements, total_output_elements, false);
    initialize_random(cpu_mem.h_matrices, total_input_elements);

    // ===== Run CPU benchmark with power measurement =====
    PowerSampler cpu_sampler;
    cpu_sampler.start();
    results.cpu_ms = run_cpu_benchmark(cpu_mem, config);
    results.cpu_power_w = cpu_sampler.stop();

    // Wait for power to stabilize
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // ===== Run OMP benchmark with power measurement =====
    PowerSampler omp_sampler;
    omp_sampler.start();
    results.omp_ms = run_omp_benchmark(cpu_mem, config);
    results.omp_power_w = omp_sampler.stop();

    // Wait for power to stabilize
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Calculate CPU GFLOPS
    long long total_flops = (long long)config.num_ops * (config.num_joints - 1) * 128;
    results.cpu_gflops = total_flops / (results.cpu_ms * 1e6);
    results.omp_gflops = total_flops / (results.omp_ms * 1e6);

    // ===== Run GPU Pinned benchmark with power measurement =====
    BenchmarkConfig pinned_config = config;
    pinned_config.use_managed = false;

    PowerSampler gpu_pinned_sampler;
    gpu_pinned_sampler.start();
    GPUTiming gpu_pinned_timing = run_gpu_benchmark(cpu_mem, pinned_config, input_bytes, output_bytes);
    results.gpu_pinned.power_w = gpu_pinned_sampler.stop();

    results.gpu_pinned.kernel_ms = gpu_pinned_timing.kernel_ms;
    results.gpu_pinned.transfer_ms = gpu_pinned_timing.transfer_ms;
    results.gpu_pinned.total_ms = gpu_pinned_timing.total_ms;
    results.gpu_pinned.gflops = total_flops / (gpu_pinned_timing.kernel_ms * 1e6);

    // Calculate speedup (using pinned as reference)
    results.speedup = results.cpu_ms / results.gpu_pinned.total_ms;

    // Verify correctness
    float base_tolerance = 1e-3f;
    float n = (config.num_joints - 2);
    float gpu_tolerance = base_tolerance * (1.0f + 0.1f * n + 0.01f * n * n);
    bool correct_cpu_omp = compare_results(cpu_mem.h_out_cpu, cpu_mem.h_out_cpu_omp, total_output_elements, 1e-5f);
    bool correct_gpu_pinned = compare_results(cpu_mem.h_out_cpu, cpu_mem.h_out_gpu, total_output_elements, gpu_tolerance);
    results.correct = correct_cpu_omp && correct_gpu_pinned;

    // ===== Run GPU Managed benchmark if unified memory available =====
    if (has_unified) {
        // Wait for power to stabilize
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        // Allocate managed memory
        MemoryPointers managed_mem = allocate_memory(total_input_elements, total_output_elements, true);

        // Copy input data from cpu_mem to managed_mem
        memcpy(managed_mem.h_matrices, cpu_mem.h_matrices, input_bytes);

        BenchmarkConfig managed_config = config;
        managed_config.use_managed = true;

        PowerSampler gpu_managed_sampler;
        gpu_managed_sampler.start();
        GPUTiming gpu_managed_timing = run_gpu_benchmark(managed_mem, managed_config, input_bytes, output_bytes);
        results.gpu_managed.power_w = gpu_managed_sampler.stop();

        results.gpu_managed.kernel_ms = gpu_managed_timing.kernel_ms;
        results.gpu_managed.transfer_ms = gpu_managed_timing.transfer_ms;
        results.gpu_managed.total_ms = gpu_managed_timing.total_ms;
        results.gpu_managed.gflops = total_flops / (gpu_managed_timing.kernel_ms * 1e6);
        results.gpu_managed.valid = true;

        // Verify managed correctness
        bool correct_gpu_managed = compare_results(cpu_mem.h_out_cpu, managed_mem.h_out_gpu, total_output_elements, gpu_tolerance);
        results.correct = results.correct && correct_gpu_managed;

        // Cleanup managed memory
        free_memory(managed_mem, true);
    }

    // Print and save results
    print_result_row(config, results, has_unified);
    append_csv_result(config, results);

    // Cleanup CPU memory
    free_memory(cpu_mem, false);
}  // ========== MAIN ==========

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

    // Test if power measurement works
    double test_power = read_jetson_power();
    if (test_power > 0.0) {
        std::cout << "Power measurement working: " << std::fixed << std::setprecision(2) << test_power << "W" << std::endl;
    } else {
        std::cerr << "[WARNING] Power measurement not available" << std::endl;
    }
    std::cout << std::endl;

    // Setup configuration
    BenchmarkConfig config;
    config.num_ops = (argc > 1) ? std::atoi(argv[1]) : 500000;
    config.threads_per_block = (argc > 2) ? std::atoi(argv[2]) : 64;
    bool has_unified = has_unified_memory();

    // Clear CSV file and write header
    std::remove(csv_filename);
    write_csv_header(has_unified);

    // Test different numbers of joints
    int joint_configs[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32};
    int num_configs = sizeof(joint_configs) / sizeof(joint_configs[0]);

    // Print header
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Matrix Chain Multiplication Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    print_header(config, has_unified);

    // Run benchmarks for each joint configuration
    for (int i = 0; i < num_configs; i++) {
        config.num_joints = joint_configs[i];
        test_num_joints(config, has_unified);
    }

    // Print legend
    print_legend();

    return 0;
}
