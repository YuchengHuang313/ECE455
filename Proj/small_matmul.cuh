#ifndef SMALL_MATMUL_CUH
#define SMALL_MATMUL_CUH

#ifdef __CUDACC__
// Device function for single 4x4 matrix multiplication
__device__ __forceinline__ void mul4x4_one(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C);

// Kernel for batched 4x4 matrix multiplication: computes (A×B)×(C×D) for each row
__global__ void small_matmul_batched(const float* A, const float* B, const float* C, const float* D, float* out, int num_rows);
#else
// Forward declaration for host code
extern "C" {
void small_matmul_batched_launch(const float* A, const float* B, const float* C, const float* D, float* out, int num_rows, int blocks_y,
                                 int threads_y);
}
#endif

#endif  // SMALL_MATMUL_CUH