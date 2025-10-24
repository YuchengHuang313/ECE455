# Portable CUDA Matrix Multiplication Benchmark

Auto-detecting CUDA benchmark for batched 4×4 matrix multiplication across CPU, OpenMP, and CUDA GPU. Designed to be portable (desktop and Jetson), reproducible, and easy to extend.

## Features

✅ **Auto-detects CPU architecture** (x86_64 vs ARM/aarch64)  
✅ **Auto-detects GPU compute capability** (SM) via `nvidia-smi`  
✅ **Cross-platform**: Works on desktop GPUs and Jetson devices  
✅ **Compares CPU vs OpenMP vs GPU** performance  

## Quick Start

```bash
# Build (auto-detects everything)
make

# Run benchmark
make run              # 1,000 matrices (default)
make run-small        # 100 matrices
make run-medium       # 10,000 matrices
make run-large        # 100,000 matrices

# Custom size
./small_matmul_test 1000000

# Show system info
make info
```

## Manual Overrides

```bash
# Force specific GPU compute capability
make SM=75            # Turing (RTX 20xx)
make SM=86            # Ampere (RTX 30xx)
make SM=87            # Jetson Orin
make SM=89            # Ada Lovelace (RTX 40xx)
```

## GPU Compute Capabilities (SM)

| SM | Architecture | GPUs |
|----|--------------|------|
| 75 | Turing | RTX 2060/2070/2080, GTX 1650/1660 |
| 86 | Ampere | RTX 3060/3070/3080/3090, A100 |
| 87 | Ampere | Jetson Orin Nano/NX/AGX |
| 89 | Ada Lovelace | RTX 4060/4070/4080/4090 |

## Auto-Detection

The Makefile automatically:
1. Detects `x86_64` vs `aarch64` via `uname -m`
2. Queries GPU SM via `nvidia-smi --query-gpu=compute_cap`
3. Falls back to sensible defaults if detection fails

## Performance Tips

- **Small batches (<1K)**: CPU single-threaded is fastest
- **Medium batches (1K-100K)**: OpenMP provides good speedup
- **Large batches (100K+)**: GPU kernel dominates (if data stays on GPU)
- **Memory transfer overhead**: Dominates GPU total time for small workloads

## Build from Scratch

```bash
make clean
make rebuild
```

## Requirements

- CUDA Toolkit (12.0+)
- C++11 compiler
- OpenMP support (`-fopenmp`)
- `nvidia-smi` (for auto-detection, optional)

## Motivation
- In robotic applications such as articulated arm or legged robot control, forward and inverse kinematics rely heavily on repeated small matrix multiplications to compute transformations between coordinate frames. For instance, each joint’s position and orientation are represented by 4×4 homogeneous transformation matrices, and evaluating the end-effector’s pose involves chaining these matrices together. During trajectory interpolation, thousands of forward/inverse kinematics computations must be performed per second to achieve smooth motion. These operations consist primarily of small matrix multiplications that can become computational bottlenecks on embedded CPUs. Therefore, exploring efficient, scalable methods for this operation — particularly on GPUs — is crucial for enabling real-time robotic control and motion planning. 

## Methodology

### Workload

- Operation: For each row i in a batch, compute (Aᵢ×Bᵢ)×(Cᵢ×Dᵢ) where A, B, C, D are 4×4 matrices stored in row-major order.
- FLOPs: One 4×4 matmul is 64 MACs = 128 FLOPs. We perform 3 matmuls per row, so 384 FLOPs per row and 3×128×N total.
- Data: Inputs are initialized with a fixed PRNG seed (uniform in [-1, 1]) for reproducibility.
- Layout: Each matrix is 16 contiguous floats; the i-th matrices start at offset i×16 in their respective arrays.

### Implementations

1) CPU (single-thread)
- Straightforward triple-loop 4×4 multiply used as a building block.
- Batched evaluation iterates rows sequentially and materializes intermediate products (A×B) and (C×D) on the stack per row.

2) CPU (OpenMP)
- Identical math to the single-thread baseline, parallelized with `#pragma omp parallel for` over rows.
- Each thread computes its own intermediates on the stack to avoid synchronization and false sharing.

3) CUDA GPU
- One thread processes one row (i.e., the full (A×B)×(C×D) chain for that row).
- Kernel uses a fully unrolled 4×4 routine (`__device__ __forceinline__`) to keep all temporaries in registers; no shared memory is required for this small problem size.
- Launch configuration: a 1D thread arrangement along y; `threadsPerBlock = 64` by default and `numBlocks = ceil(N / threadsPerBlock)`.
- Memory access: Each thread reads 16 floats from A, B, C, and D at contiguous offsets; across threads these reads are strided by 16 floats, which yields coalesced 32B/128B transactions on modern GPUs when arrays are aligned.

### Build and Auto-Detection

- The Makefile probes CPU ISA via `uname -m` (x86_64 vs aarch64) and queries GPU compute capability (SM) via `nvidia-smi`.
- Flags are set accordingly (e.g., `-arch=sm_${SM}` for `nvcc`, `-fopenmp` for CPU OpenMP).
- Users can override detection with `make SM=86` (see Manual Overrides).

### Timing and Metrics

- CPU timers: `std::chrono::high_resolution_clock` around the full batched loops for both single-thread and OpenMP.
- GPU timers: host-side wall clock split into three segments with `cudaDeviceSynchronize()` to bound the kernel portion:
	- Copy host→device
	- Kernel execution
	- Copy device→host
- Total GPU time includes transfers; “kernel-only” time excludes transfers. This highlights the transfer overhead for small batches.
- Throughput (GFLOPS) is computed as total_FLOPs / elapsed_seconds / 1e9, with total_FLOPs = 3×128×N.
- Note: CUDA events (`cudaEventRecord`) would provide finer-grained device-side kernel timing; host-side timing with synchronization is sufficient for this benchmark’s comparisons.

### Correctness Verification

- We compare GPU results against CPU single-thread output element-wise with a tolerance of 1e-4.
- To reduce host memory usage on very large runs, we retain up to 10,000 elements from the CPU result for verification while freeing the full buffer.
- Any mismatch reports the first differing index and values to aid debugging.

### Experiment Protocol

- Batch sizes: run provided presets (`run-small`, `run`, `run-medium`, `run-large`) to observe scaling across regimes where CPU, OpenMP, or GPU dominate.
- Threading: OpenMP uses `omp_get_max_threads()` by default; you can control threads via `OMP_NUM_THREADS`.
- Warm-up: No explicit warm-up is performed; on systems with cold GPU clocks, consider an initial throwaway run for more stable numbers.
- Repeats: For publication-quality numbers, average multiple runs and report mean ± stdev.

### Interpreting Results

- Small batches (<1K): kernel launch and PCIe transfers dominate; CPU often wins.
- Medium batches (1K–100K): OpenMP typically provides a strong speedup over single-thread and may compete with GPU depending on transfer cost.
- Large batches (≥100K): GPU kernel time dominates overall and generally yields the best throughput if data residency on the device is maintained.

## Reproducibility

- Deterministic inputs: A fixed PRNG seed is used to initialize matrices for repeatable results.
- System snapshot: Use the included target to log system info alongside results.

```bash
# Capture system info used during the run
make info | tee system_info.txt

# Optional: capture compiler and driver details
nvcc --version | tee -a system_info.txt
nvidia-smi -q | tee -a system_info.txt
uname -a | tee -a system_info.txt
```

- Threading and affinity: Control OpenMP parallelism with `OMP_NUM_THREADS` and consider pinning threads (e.g., `GOMP_CPU_AFFINITY` on GCC).
- Run protocol: For stable numbers, perform a short warm-up run, then take the mean ± stdev over several repetitions.

## Limitations and Future Work

- Timing granularity: Host-side timing with `cudaDeviceSynchronize()` is sufficient for high-level comparisons but not cycle-accurate. Future: use CUDA events for device-side kernel timing and report both.
- Transfer overhead: For small batches, PCIe transfers dominate. Future: use pinned (page-locked) memory and CUDA streams to overlap H2D/D2H with compute; consider keeping data resident on the GPU for iterative workloads.
- Kernel launch overhead: Very small N is sensitive to launch latency. Future: CUDA Graphs can reduce launch overhead.
- Memory layout: Current layout is row-major batches of 4×4. Future: experiment with structure-of-arrays (SoA) or interleaving to improve coalescing on large N.
- Math libraries: Add a cuBLAS strided batched SGEMM baseline for reference on small GEMMs.
- Tensor cores: Explore WMMA/CUTLASS packing strategies to leverage tensor cores for 4×4 blocks when applicable.
- CPU vectorization: Add explicit SIMD (e.g., AVX2/AVX-512/NEON) and compiler flags (`-O3 -march=native`) with microbenchmarks.

## Troubleshooting

- `nvidia-smi: command not found` or no devices shown: Ensure the NVIDIA driver is installed and the host sees the GPU. On Jetson, `nvidia-smi` may be limited; set `SM` manually (see Manual Overrides).
- `nvcc fatal: Unsupported gpu architecture 'sm_XY'`: Choose a supported SM for your CUDA Toolkit or upgrade CUDA.
- OpenMP not enabled: Make sure your compiler supports `-fopenmp` and your toolchain links the OpenMP runtime.
- Runtime mismatches in verification: Verify drivers and toolkit versions; try a slightly larger tolerance (e.g., 1e-3) to account for different math orders on some platforms.
- Out-of-memory for huge runs: Reduce N or run the preset sizes; GPU memory must accommodate all four input batches and one output batch.

## Project Layout

- `small_matmul.cpp`: Host benchmark harness (CPU, OpenMP, and CUDA integration, timing, verification, reporting)
- `small_matmul.cu`: CUDA kernel and launch wrapper for batched 4×4 matmul
- `README.md`: This document

If you want a cuBLAS baseline or event-based timing, open an issue or add a PR—both are straightforward extensions to this setup.
