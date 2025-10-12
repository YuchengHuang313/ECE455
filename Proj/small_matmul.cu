#include <cuda_runtime.h>

#include "small_matmul.cuh"

#define MAT_SIZE 4

__global__ void small_matmul_batched(const float* A, const float* B, const float* C, const float* D, float* out, int num_rows) {
    // this would only be called with a vertical block
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= num_rows) return;

    // extract A B C D
    float result_AB[MAT_SIZE * MAT_SIZE];
    mul4x4_one(&A[row * MAT_SIZE * MAT_SIZE], &B[row * MAT_SIZE * MAT_SIZE], result_AB);

    float result_CD[MAT_SIZE * MAT_SIZE];
    mul4x4_one(&C[row * MAT_SIZE * MAT_SIZE], &D[row * MAT_SIZE * MAT_SIZE], result_CD);

    // find final result matrix - write to this thread's output location
    mul4x4_one(result_AB, result_CD, &out[row * MAT_SIZE * MAT_SIZE]);
}

__device__ __forceinline__ void mul4x4_one(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    // load rows of A
    float a00 = A[0], a01 = A[1], a02 = A[2], a03 = A[3];
    float a10 = A[4], a11 = A[5], a12 = A[6], a13 = A[7];
    float a20 = A[8], a21 = A[9], a22 = A[10], a23 = A[11];
    float a30 = A[12], a31 = A[13], a32 = A[14], a33 = A[15];

    // prefetch B columns (still row-major; step by 4 for columns)
    float b00 = B[0], b01 = B[1], b02 = B[2], b03 = B[3];
    float b10 = B[4], b11 = B[5], b12 = B[6], b13 = B[7];
    float b20 = B[8], b21 = B[9], b22 = B[10], b23 = B[11];
    float b30 = B[12], b31 = B[13], b32 = B[14], b33 = B[15];

    // row 0
    C[0] = a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30;
    C[1] = a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31;
    C[2] = a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32;
    C[3] = a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33;
    // row 1
    C[4] = a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30;
    C[5] = a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31;
    C[6] = a10 * b02 + a11 * b12 + a12 * b22 + a13 * b32;
    C[7] = a10 * b03 + a11 * b13 + a12 * b23 + a13 * b33;
    // row 2
    C[8] = a20 * b00 + a21 * b10 + a22 * b20 + a23 * b30;
    C[9] = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31;
    C[10] = a20 * b02 + a21 * b12 + a22 * b22 + a23 * b32;
    C[11] = a20 * b03 + a21 * b13 + a22 * b23 + a23 * b33;
    // row 3
    C[12] = a30 * b00 + a31 * b10 + a32 * b20 + a33 * b30;
    C[13] = a30 * b01 + a31 * b11 + a32 * b21 + a33 * b31;
    C[14] = a30 * b02 + a31 * b12 + a32 * b22 + a33 * b32;
    C[15] = a30 * b03 + a31 * b13 + a32 * b23 + a33 * b33;
}

// C-linkage wrapper for launching the kernel from C++ code
extern "C" void small_matmul_batched_launch(const float* A, const float* B, const float* C, const float* D, float* out, int num_rows, int blocks_y,
                                            int threads_y) {
    dim3 blocks(1, blocks_y);
    dim3 threads(1, threads_y);
    small_matmul_batched<<<blocks, threads>>>(A, B, C, D, out, num_rows);
}
