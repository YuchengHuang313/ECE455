#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <iostream>

#include "small_matmul.cuh"

#define MAT_SIZE 4

// Check CUDA errors
#define CUDA_CHECK(call)                                                                                                   \
    do {                                                                                                                   \
        cudaError_t err = call;                                                                                            \
        if (err != cudaSuccess) {                                                                                          \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                                                                            \
        }                                                                                                                  \
    } while (0)

void print_matrix(const float* mat, const char* name) {
    std::cout << name << ":\n";
    for (int i = 0; i < MAT_SIZE; i++) {
        for (int j = 0; j < MAT_SIZE; j++) {
            std::cout << mat[i * MAT_SIZE + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // Test parameters
    int num_rows = 10;
    int num_joints = 4;

    if (argc > 1) {
        num_rows = std::atoi(argv[1]);
    }
    if (argc > 2) {
        num_joints = std::atoi(argv[2]);
    }

    std::cout << "Testing small_matmul_batched_combined" << std::endl;
    std::cout << "Number of rows: " << num_rows << std::endl;
    std::cout << "Number of joints per row: " << num_joints << std::endl;
    std::cout << std::endl;

    // Calculate sizes
    const int mat_size = MAT_SIZE * MAT_SIZE;
    const int total_input_elements = num_rows * num_joints * mat_size;
    const int total_output_elements = num_rows * mat_size;
    const size_t input_bytes = total_input_elements * sizeof(float);
    const size_t output_bytes = total_output_elements * sizeof(float);

    // Allocate host memory
    float* h_matrices = new float[total_input_elements];
    float* h_out_cpu = new float[total_output_elements];
    float* h_out_gpu = new float[total_output_elements];

    // Initialize with random data
    std::cout << "Initializing random data..." << std::endl;
    initialize_random(h_matrices, total_input_elements);

    // Run CPU version
    std::cout << "Running CPU reference implementation..." << std::endl;
    small_matmul_batched_combined_cpu(h_matrices, h_out_cpu, num_rows, num_joints);

    // Allocate device memory
    float *d_matrices, *d_out;
    CUDA_CHECK(cudaMalloc(&d_matrices, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, output_bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_matrices, h_matrices, input_bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    std::cout << "Running GPU kernel..." << std::endl;
    const int threadsPerBlock = 64;
    int numBlocks = (num_rows + threadsPerBlock - 1) / threadsPerBlock;

    dim3 blocks(numBlocks, 1);
    dim3 threads(threadsPerBlock, 1);
    small_matmul_batched_combined<<<blocks, threads>>>(d_matrices, d_out, num_rows, num_joints);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, output_bytes, cudaMemcpyDeviceToHost));

    // Verify results
    std::cout << "\nVerifying results..." << std::endl;
    bool correct = compare_results(h_out_cpu, h_out_gpu, total_output_elements, 1e-3f);

    if (correct) {
        std::cout << "✓ TEST PASSED: GPU results match CPU!" << std::endl;
    } else {
        std::cout << "✗ TEST FAILED: GPU results DO NOT match CPU!" << std::endl;

        // Print first mismatch for debugging
        std::cout << "\nFirst result (row 0):" << std::endl;
        print_matrix(h_out_cpu, "CPU");
        print_matrix(h_out_gpu, "GPU");
    }

    // Cleanup
    delete[] h_matrices;
    delete[] h_out_cpu;
    delete[] h_out_gpu;
    CUDA_CHECK(cudaFree(d_matrices));
    CUDA_CHECK(cudaFree(d_out));

    return correct ? 0 : 1;
}
