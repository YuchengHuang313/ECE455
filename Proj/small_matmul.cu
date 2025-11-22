#include <cuda_runtime.h>
#include <omp.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include "small_matmul.cuh"

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

__global__ void small_matmul_batched_combined(const float* matrix, float* out, int num_rows, int num_joints) {
    // this would only be called with a vertical block
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    // matrix contains arbitrary number of 4x4 matrices as a 16x1 vector per row
    // there might be an arbitrary number of joints, so we could have A, B, C, D, E, F, ...
    // we will multiply them all together in sequence: (((A×B)×C)×D)...
    // each thread calculates matrix mult in sequence
    float result[MAT_SIZE * MAT_SIZE];
    // Initialize result to identity matrix

#pragma unroll
    for (int i = 0; i < MAT_SIZE * MAT_SIZE; i++) {
        result[i] = (i % (MAT_SIZE + 1) == 0) ? 1.0f : 0.0f;
    }

#pragma unroll
    for (int joint = 0; joint < num_joints; joint++) {
        float temp[MAT_SIZE * MAT_SIZE];
        mul4x4_one(result, &matrix[(row * num_joints + joint) * MAT_SIZE * MAT_SIZE], temp);
        // Copy temp back to result for next iteration
        for (int i = 0; i < MAT_SIZE * MAT_SIZE; i++) {
            result[i] = temp[i];
        }
    }

    // write final result to output
#pragma unroll
    for (int i = 0; i < MAT_SIZE * MAT_SIZE; i++) {
        out[row * MAT_SIZE * MAT_SIZE + i] = result[i];
    }
}
// ========== CPU FUNCTIONS ==========

// CPU version of 4x4 matrix multiplication - optimized
void mul4x4_cpu(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    // Unrolled version for 4x4 - similar to GPU device function
    // Load A rows
    const float a00 = A[0], a01 = A[1], a02 = A[2], a03 = A[3];
    const float a10 = A[4], a11 = A[5], a12 = A[6], a13 = A[7];
    const float a20 = A[8], a21 = A[9], a22 = A[10], a23 = A[11];
    const float a30 = A[12], a31 = A[13], a32 = A[14], a33 = A[15];

    // Load B columns (from row-major)
    const float b00 = B[0], b01 = B[1], b02 = B[2], b03 = B[3];
    const float b10 = B[4], b11 = B[5], b12 = B[6], b13 = B[7];
    const float b20 = B[8], b21 = B[9], b22 = B[10], b23 = B[11];
    const float b30 = B[12], b31 = B[13], b32 = B[14], b33 = B[15];

    // Row 0
    C[0] = a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30;
    C[1] = a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31;
    C[2] = a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32;
    C[3] = a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33;

    // Row 1
    C[4] = a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30;
    C[5] = a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31;
    C[6] = a10 * b02 + a11 * b12 + a12 * b22 + a13 * b32;
    C[7] = a10 * b03 + a11 * b13 + a12 * b23 + a13 * b33;

    // Row 2
    C[8] = a20 * b00 + a21 * b10 + a22 * b20 + a23 * b30;
    C[9] = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31;
    C[10] = a20 * b02 + a21 * b12 + a22 * b22 + a23 * b32;
    C[11] = a20 * b03 + a21 * b13 + a22 * b23 + a23 * b33;

    // Row 3
    C[12] = a30 * b00 + a31 * b10 + a32 * b20 + a33 * b30;
    C[13] = a30 * b01 + a31 * b11 + a32 * b21 + a33 * b31;
    C[14] = a30 * b02 + a31 * b12 + a32 * b22 + a33 * b32;
    C[15] = a30 * b03 + a31 * b13 + a32 * b23 + a33 * b33;
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
void small_matmul_batched_cpu_omp(const float* __restrict__ A, const float* __restrict__ B, const float* __restrict__ C, const float* __restrict__ D,
                                  float* __restrict__ out, int num_rows) {
    // Use static schedule with optimal chunk size for cache locality
    // Empirically tested: chunk size 256-4096 performs best (4096 gives ~53 GFLOPS)
    // Smaller chunks (256) also work well, larger chunks (4096) slightly better
#pragma omp parallel for schedule(static, 4096)
    for (int row = 0; row < num_rows; row++) {
        // Stack-allocated temporary arrays for intermediate results
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

// CPU version of combined batched matrix multiplication
void small_matmul_batched_combined_cpu(const float* matrix, float* out, int num_rows, int num_joints) {
    const int mat_size = MAT_SIZE * MAT_SIZE;

    for (int row = 0; row < num_rows; row++) {
        const float* base = matrix + row * num_joints * mat_size;
        float result[mat_size];

        // Initialize result to identity matrix
        for (int i = 0; i < mat_size; i++) {
            result[i] = (i % (MAT_SIZE + 1) == 0) ? 1.0f : 0.0f;
        }

        // Chain multiply all matrices: I × M0 × M1 × M2 × ...
        for (int joint = 0; joint < num_joints; joint++) {
            float temp[mat_size];
            mul4x4_cpu(result, base + joint * mat_size, temp);
            // Copy temp back to result for next iteration
            for (int i = 0; i < mat_size; i++) {
                result[i] = temp[i];
            }
        }

        // Copy final result to output
        for (int i = 0; i < mat_size; i++) {
            out[row * mat_size + i] = result[i];
        }
    }
}

// OpenMP parallelized version of combined batched matrix multiplication
void small_matmul_batched_combined_cpu_omp(const float* __restrict__ matrix, float* __restrict__ out, int num_rows, int num_joints) {
    const int mat_size = MAT_SIZE * MAT_SIZE;

#pragma omp parallel for schedule(static, 4096)
    for (int row = 0; row < num_rows; row++) {
        const float* base = matrix + row * num_joints * mat_size;
        float result[mat_size];

        // Initialize result to identity matrix
        for (int i = 0; i < mat_size; i++) {
            result[i] = (i % (MAT_SIZE + 1) == 0) ? 1.0f : 0.0f;
        }

        // Chain multiply all matrices: I × M0 × M1 × M2 × ...
        for (int joint = 0; joint < num_joints; joint++) {
            float temp[mat_size];
            mul4x4_cpu(result, base + joint * mat_size, temp);
            // Copy temp back to result for next iteration
            for (int i = 0; i < mat_size; i++) {
                result[i] = temp[i];
            }
        }

        // Copy final result to output
        for (int i = 0; i < mat_size; i++) {
            out[row * mat_size + i] = result[i];
        }
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
            std::cout << "Mismatch at index " << i << ": CPU=" << cpu_result[i] << ", GPU=" << gpu_result[i] << std::scientific
                      << std::setprecision(6) << ", diff=" << diff << std::defaultfloat << std::setprecision(6) << " (tol=" << tolerance << ")"
                      << std::endl;
            return false;
        }
    }
    return true;
}
