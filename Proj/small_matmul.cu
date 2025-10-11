#include <cuda_runtime.h>

#include "small_matmul.cuh"

// Each thread handles one complete row (one set of 4x4 matrix multiplications)
// Computes D = (A * B) * C for one batch
__global__ void small_matmul_batched(const float* A, const float* B, const float* C, float* D, const int num_rows, const int num_cols) {
    // Global thread ID - each thread processes one row
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int inner_n = std::sqrt(num_cols); // Since we are dealing with 4x4 matrices
    if (row >= num_rows || col >= num_cols) return;

    // Pointers to the current row's flattened 4x4 matrices
    const float* matA = A + row * num_cols;  // 16 elements
    const float* matB = B + row * num_cols;  // 16 elements
    const float* matC = C + row * num_cols;  // 16 elements
    float* matD = D + row * num_cols;        // 16 elements output

    // Temporary storage for intermediate result (A * B)
    float temp[16];
    // First compute temp = A * B
    float val = 0.0f;
    for (int i = 0; i < inner_n; ++i) {
        val += matA[col * inner_n + i] * matB[i * inner_n + col];
    }
    temp[col] = val;
    __syncthreads();
    // Then compute D = temp * C
    val = 0.0f;
    for (int i = 0; i < inner_n; ++i) {
        val += temp[col * inner_n + i] * matC[i * inner_n + col];
    }
    matD[col] = val;


}

__device__ void vec_matmul(const float* A, const float* B, float* C, const int inner_n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    float val = 0.0f;
    for (int i = 0; i < inner_n; ++i) {
        val += A[row * inner_n + i] * B[i * inner_n + col];
    }
    C[row * inner_n + col] = val;
}