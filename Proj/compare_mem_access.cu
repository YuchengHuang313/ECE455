#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>

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
        std::cout << "Usage: " << argv[0] << " [size_mb]" << std::endl;
        std::cout << "\nCompares transfer performance between pageable (new[]) and pinned (cudaHostAlloc) memory" << std::endl;
        std::cout << "\nArguments:" << std::endl;
        std::cout << "  size_mb  Size of data to transfer in MB (default: 256)" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  " << argv[0] << " 512" << std::endl;
        return 0;
    }

    // Default size: 256 MB
    int size_mb = 256;
    if (argc > 1) {
        size_mb = std::atoi(argv[1]);
    }

    // Use size_t to avoid overflow for large allocations
    const size_t bytes = static_cast<size_t>(size_mb) * 1024ULL * 1024ULL;
    const size_t elements = bytes / sizeof(float);

    std::cout << "========================================" << std::endl;
    std::cout << "  Memory Allocation Transfer Test" << std::endl;
    std::cout << "  Data size: " << size_mb << " MB (" << std::fixed << std::setprecision(2) << bytes / (1024.0 * 1024.0) << " MB)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Allocate pageable memory (standard new[])
    float* h_pageable = new float[elements];

    // Allocate pinned memory (CUDA page-locked)
    float* h_pinned;
    CUDA_CHECK(cudaHostAlloc(&h_pinned, bytes, cudaHostAllocDefault));

    // Allocate device memory
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    // Initialize both buffers with same data
    for (size_t i = 0; i < elements; i++) {
        h_pageable[i] = static_cast<float>(i % 100);
    }
    std::memcpy(h_pinned, h_pageable, bytes);

    std::cout << "Testing pageable memory (new[])..." << std::endl;

    // ========== PAGEABLE MEMORY TRANSFERS ==========
    // Host to Device
    auto pageable_h2d_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_data, h_pageable, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto pageable_h2d_end = std::chrono::high_resolution_clock::now();

    // Device to Host
    auto pageable_d2h_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(h_pageable, d_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto pageable_d2h_end = std::chrono::high_resolution_clock::now();

    auto pageable_h2d_us = std::chrono::duration_cast<std::chrono::microseconds>(pageable_h2d_end - pageable_h2d_start).count();
    auto pageable_d2h_us = std::chrono::duration_cast<std::chrono::microseconds>(pageable_d2h_end - pageable_d2h_start).count();
    double pageable_h2d_bw = (bytes / (1024.0 * 1024.0 * 1024.0)) / (pageable_h2d_us / 1e6);
    double pageable_d2h_bw = (bytes / (1024.0 * 1024.0 * 1024.0)) / (pageable_d2h_us / 1e6);

    std::cout << "Testing pinned memory (cudaHostAlloc)..." << std::endl;

    // ========== PINNED MEMORY TRANSFERS ==========
    // Host to Device
    auto pinned_h2d_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto pinned_h2d_end = std::chrono::high_resolution_clock::now();

    // Device to Host
    auto pinned_d2h_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(h_pinned, d_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto pinned_d2h_end = std::chrono::high_resolution_clock::now();

    auto pinned_h2d_us = std::chrono::duration_cast<std::chrono::microseconds>(pinned_h2d_end - pinned_h2d_start).count();
    auto pinned_d2h_us = std::chrono::duration_cast<std::chrono::microseconds>(pinned_d2h_end - pinned_d2h_start).count();
    double pinned_h2d_bw = (bytes / (1024.0 * 1024.0 * 1024.0)) / (pinned_h2d_us / 1e6);
    double pinned_d2h_bw = (bytes / (1024.0 * 1024.0 * 1024.0)) / (pinned_d2h_us / 1e6);

    // ========== PRINT RESULTS ==========
    std::cout << std::endl;
    std::cout << std::setw(25) << "Memory Type" << " | " << std::setw(14) << "H→D (ms)" << " | " << std::setw(14) << "H→D (GB/s)"
              << " | " << std::setw(14) << "D→H (ms)" << " | " << std::setw(12) << "D→H (GB/s)" << std::endl;
    std::cout << "--------------------------|--------------|--------------|--------------|-------------" << std::endl;

    std::cout << std::setw(25) << "Pageable (new[])" << " | " << std::setw(12) << std::fixed << std::setprecision(2) << pageable_h2d_us / 1000.0
              << " | " << std::setw(12) << pageable_h2d_bw << " | " << std::setw(12) << pageable_d2h_us / 1000.0 << " | " << std::setw(12)
              << pageable_d2h_bw << std::endl;

    std::cout << std::setw(25) << "Pinned (cudaHostAlloc)" << " | " << std::setw(12) << pinned_h2d_us / 1000.0 << " | " << std::setw(12)
              << pinned_h2d_bw << " | " << std::setw(12) << pinned_d2h_us / 1000.0 << " | " << std::setw(12) << pinned_d2h_bw << std::endl;

    std::cout << std::endl;
    std::cout << "Speedup (Pinned vs Pageable):" << std::endl;
    std::cout << "  H→D: " << std::setprecision(2) << (double)pageable_h2d_us / pinned_h2d_us << "x faster" << std::endl;
    std::cout << "  D→H: " << (double)pageable_d2h_us / pinned_d2h_us << "x faster" << std::endl;

    std::cout << std::endl;
    std::cout << "Time saved (Pinned vs Pageable):" << std::endl;
    std::cout << "  H→D: " << (pageable_h2d_us - pinned_h2d_us) / 1000.0 << " ms saved" << std::endl;
    std::cout << "  D→H: " << (pageable_d2h_us - pinned_d2h_us) / 1000.0 << " ms saved" << std::endl;
    std::cout << "  Total: " << ((pageable_h2d_us + pageable_d2h_us) - (pinned_h2d_us + pinned_d2h_us)) / 1000.0 << " ms saved per round-trip"
              << std::endl;

    std::cout << std::endl;
    std::cout << "Legend:" << std::endl;
    std::cout << "  Pageable  = Standard C++ allocation (new[]) - OS can page/swap" << std::endl;
    std::cout << "  Pinned    = CUDA page-locked memory - fixed physical address" << std::endl;
    std::cout << "  H→D       = Host to Device transfer" << std::endl;
    std::cout << "  D→H       = Device to Host transfer" << std::endl;
    std::cout << "========================================" << std::endl;

    // Cleanup
    delete[] h_pageable;
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}
