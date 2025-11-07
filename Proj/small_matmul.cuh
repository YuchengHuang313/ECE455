#ifndef SMALL_MATMUL_CUH
#define SMALL_MATMUL_CUH

#include <cuda_runtime.h>

// Device function for single 4x4 matrix multiplication
__device__ __forceinline__ void mul4x4_one(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C);

// Kernel for batched 4x4 matrix multiplication: computes (A×B)×(C×D) for each row
__global__ void small_matmul_batched(const float* A, const float* B, const float* C, const float* D, float* out, int num_rows);

// CPU functions
void mul4x4_cpu(const float* A, const float* B, float* C);
void small_matmul_batched_cpu(const float* A, const float* B, const float* C, const float* D, float* out, int num_rows);
void small_matmul_batched_cpu_omp(const float* A, const float* B, const float* C, const float* D, float* out, int num_rows);

// Helper functions
void initialize_random(float* data, int size);
bool compare_results(const float* cpu_result, const float* gpu_result, int size, float tolerance);

#endif  // SMALL_MATMUL_CUH