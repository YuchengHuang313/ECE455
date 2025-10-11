#ifndef SMALL_MATMUL_CUH
#define SMALL_MATMUL_CUH

__global__ void small_matmul_batched(const float* A, const float* B, const float* C, float* D, int N);

#endif // SMALL_MATMUL_CUH