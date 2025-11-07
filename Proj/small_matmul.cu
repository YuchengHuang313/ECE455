#include "small_matmul.cuh"

#include <cuda_runtime.h>
#include <omp.h>

#include <cmath>
#include <iostream>
#include <random>

#define MAT_SIZE 4

// ========== CUDA KERNEL AND DEVICE FUNCTIONS ==========

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

__global__ void small_matmul_batched(const float* A, const float* B, const float* C, const float* D, float* out, int num_rows) {
    // this would only be called with a vertical block
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    // extract A B C D
    float result_AB[MAT_SIZE * MAT_SIZE];
    mul4x4_one(&A[row * MAT_SIZE * MAT_SIZE], &B[row * MAT_SIZE * MAT_SIZE], result_AB);

    float result_CD[MAT_SIZE * MAT_SIZE];
    mul4x4_one(&C[row * MAT_SIZE * MAT_SIZE], &D[row * MAT_SIZE * MAT_SIZE], result_CD);

    // find final result matrix - write to this thread's output location
    mul4x4_one(result_AB, result_CD, &out[row * MAT_SIZE * MAT_SIZE]);
}

// ========== CPU FUNCTIONS ==========

// CPU version of 4x4 matrix multiplication
void mul4x4_cpu(const float* A, const float* B, float* C) {
    for (int i = 0; i < MAT_SIZE; i++) {
        for (int j = 0; j < MAT_SIZE; j++) {
            float sum = 0.0f;
            for (int k = 0; k < MAT_SIZE; k++) {
                sum += A[i * MAT_SIZE + k] * B[k * MAT_SIZE + j];
            }
            C[i * MAT_SIZE + j] = sum;
        }
    }
}

// CPU version of batched matrix multiplication
void small_matmul_batched_cpu(const float* A, const float* B, const float* C, const float* D, float* out, int num_rows) {
    for (int row = 0; row < num_rows; row++) {
        float result_AB[MAT_SIZE * MAT_SIZE];
        float result_CD[MAT_SIZE * MAT_SIZE];

        // Compute A×B
        mul4x4_cpu(&A[row * MAT_SIZE * MAT_SIZE], &B[row * MAT_SIZE * MAT_SIZE], result_AB);

        // Compute C×D
        mul4x4_cpu(&C[row * MAT_SIZE * MAT_SIZE], &D[row * MAT_SIZE * MAT_SIZE], result_CD);

        // Compute (A×B)×(C×D)
        mul4x4_cpu(result_AB, result_CD, &out[row * MAT_SIZE * MAT_SIZE]);
    }
}

// OpenMP parallelized version of batched matrix multiplication
void small_matmul_batched_cpu_omp(const float* A, const float* B, const float* C, const float* D, float* out, int num_rows) {
#pragma omp parallel for schedule(static)
    for (int row = 0; row < num_rows; row++) {
        float result_AB[MAT_SIZE * MAT_SIZE];
        float result_CD[MAT_SIZE * MAT_SIZE];

        // Compute A×B
        mul4x4_cpu(&A[row * MAT_SIZE * MAT_SIZE], &B[row * MAT_SIZE * MAT_SIZE], result_AB);

        // Compute C×D
        mul4x4_cpu(&C[row * MAT_SIZE * MAT_SIZE], &D[row * MAT_SIZE * MAT_SIZE], result_CD);

        // Compute (A×B)×(C×D)
        mul4x4_cpu(result_AB, result_CD, &out[row * MAT_SIZE * MAT_SIZE]);
    }
}

// ========== HELPER FUNCTIONS ==========

// Helper function to initialize matrices with random values
void initialize_random(float* data, int size) {
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

// Helper function to compare two result arrays
bool compare_results(const float* cpu_result, const float* gpu_result, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        float diff = std::abs(cpu_result[i] - gpu_result[i]);
        if (diff > tolerance) {
            std::cout << "Mismatch at index " << i << ": CPU=" << cpu_result[i] << ", GPU=" << gpu_result[i] << ", diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}
