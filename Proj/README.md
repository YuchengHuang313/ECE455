# GPU-Accelerated Robotics Matrix Multiplication: Performance Analysis

Comprehensive benchmarking suite for small batched 4×4 matrix multiplications on CPU, OpenMP, and CUDA GPU, with focus on robotic kinematics applications. This project investigates memory transfer optimizations, memory layout strategies, and scalability across varying chain lengths to enable real-time forward/inverse kinematics computations.

📄 **[View Project Poster (PDF)](./455-poster.pdf)**

## Project Overview

In robotic applications such as articulated arm or legged robot control, forward and inverse kinematics rely heavily on repeated small matrix multiplications to compute transformations between coordinate frames. Each joint's position and orientation is represented by 4×4 homogeneous transformation matrices, and evaluating the end-effector's pose involves chaining these matrices together.

This project systematically analyzes three critical aspects of GPU acceleration for robotics workloads:

1. **Memory Transfer Performance**: Comparing pageable vs. pinned vs. managed memory strategies
2. **Memory Layout Optimization**: Evaluating separate vs. interleaved matrix storage  
3. **Variable Chain Length Scaling**: Performance analysis across different kinematic chain lengths (2-32 joints)

## Benchmarks

### 1. `compare_mem_access` - Memory Transfer Analysis
Compares pageable (`new[]`), pinned (`cudaHostAlloc`), and managed (`cudaMallocManaged`) memory:
```bash
make compare_mem_access
./compare_mem_access 256  # Test with 256 MB
```
**Outputs**: H2D/D2H transfer times and bandwidth (GB/s) to CSV

### 2. `compare_mem_layout` - Memory Layout Comparison
Compares separate arrays (A, B, C, D) vs. combined interleaved layout:
```bash
make compare_mem_layout
./compare_mem_layout 500000 64  # 500K matrices, 64 threads/block
```
**Outputs**: CPU, OpenMP, GPU execution times and GFLOPS to CSV

### 3. `compare_variable_joints` - Chain Length Scaling
Tests performance across 2-32 joints (variable chain length):
```bash
make compare_variable_joints
./compare_variable_joints 500000 64  # 500K operations, 64 threads/block
```
**Outputs**: Execution time, GFLOPS, power consumption, energy efficiency to CSV

## Quick Start

```bash
# Build all benchmarks
make

# Run individual benchmarks
make run_mem_access
make run_mem_layout  
make run_joints

# View results in Jupyter notebook
python3 -m jupyter notebook memory_comparison.ipynb
```

## Visualization

The `memory_comparison.ipynb` Jupyter notebook provides comprehensive analysis:
- **Memory transfer bandwidth** comparison charts (H2D/D2H for pageable/pinned/managed)
- **Execution time vs. configuration** plots (separate vs. combined layouts)
- **GPU speedup analysis** across chain lengths (2-32 joints)
- **Kernel vs. transfer time** breakdown (stacked bar charts)
- **Power consumption** and energy efficiency metrics
- **GFLOPS per Watt** energy efficiency comparisons

## Technical Implementation

### Workload
- **Operation**: For each row i, compute (Aᵢ×Bᵢ)×(Cᵢ×Dᵢ) where A, B, C, D are 4×4 matrices
- **FLOPs**: 384 FLOPs per operation (3 matrix multiplications × 128 FLOPs each)
- **Data**: Row-major layout, 16 floats per 4×4 matrix
- **Initialization**: Fixed PRNG seed (uniform [-1, 1]) for reproducibility

### CPU Implementations
1. **Single-threaded**: Sequential triple-loop 4×4 multiply, stack-allocated intermediates
2. **OpenMP**: `#pragma omp parallel for` over rows, per-thread stack allocation to avoid false sharing

### GPU Implementation
- **Kernel**: One thread per row, fully unrolled 4×4 multiply keeping all temporaries in registers
- **Launch config**: 1D grid, 64 threads/block (default), `ceil(N/64)` blocks
- **Memory access**: Coalesced reads (16-float stride between threads)
- **Optimization**: No shared memory needed—small problem size keeps everything in registers

### Timing Methodology
- **CPU**: `std::chrono::high_resolution_clock` around batched loops
- **GPU**: Host-side wall clock with `cudaDeviceSynchronize()` segmented into:
  - Host→Device transfer
  - Kernel execution  
  - Device→Host transfer
- **Throughput**: GFLOPS = (3 × 128 × N) / elapsed_seconds / 1e9

### Correctness Verification
- Element-wise comparison: GPU vs CPU with 1e-4 tolerance
- Memory-efficient: Retains max 10K elements for verification on large runs

## System Requirements

- CUDA Toolkit (12.0+)
- C++11 compiler with OpenMP support (`-fopenmp`)
- Python 3.x with pandas, matplotlib, seaborn (for visualization)
- `nvidia-smi` (optional, for GPU auto-detection)

## Auto-Detection

The Makefile automatically:
1. Detects CPU architecture (`x86_64` vs `aarch64`) via `uname -m`
2. Queries GPU compute capability (SM) via `nvidia-smi --query-gpu=compute_cap`
3. Falls back to sensible defaults if detection fails

### Manual GPU Override

```bash
make SM=75            # Turing (RTX 20xx, GTX 16xx)
make SM=86            # Ampere (RTX 30xx, A100)
make SM=87            # Ampere (Jetson Orin Nano/NX/AGX)
make SM=89            # Ada Lovelace (RTX 40xx)
```

## Build Options

```bash
# Clean build
make clean
make

# Show detected system info
make info

# Build specific benchmarks
make compare_mem_access
make compare_mem_layout
make compare_variable_joints
```

## Project Structure

```
.
├── small_matmul.cuh              # Kernel declarations and CPU function prototypes
├── small_matmul.cu               # CUDA kernel implementations
├── compare_mem_access.cu         # Memory transfer benchmark (pageable/pinned/managed)
├── compare_mem_layout.cu         # Layout comparison (separate vs. combined)
├── compare_variable_joints.cu    # Chain length scaling analysis
├── memory_comparison.ipynb       # Visualization and analysis notebook
├── Makefile                      # Auto-detecting build system
└── README.md                     # This file
```

## Reproducibility

All benchmarks use:
- **Deterministic inputs**: Fixed PRNG seed for repeatable results
- **Multiple runs**: Default 10 runs per configuration, averaged
- **CSV outputs**: All timing data saved for post-processing

Capture system info for documentation:
```bash
make info | tee system_info.txt
nvcc --version | tee -a system_info.txt
nvidia-smi -q | tee -a system_info.txt
```

Control OpenMP threading:
```bash
export OMP_NUM_THREADS=8
./compare_variable_joints 500000 64
```

## Performance Interpretation

- **Small batches (<1K)**: Kernel launch and PCIe transfers dominate; CPU often wins
- **Medium batches (1K-100K)**: OpenMP provides strong speedup, competes with GPU
- **Large batches (≥100K)**: GPU kernel time dominates, yields best throughput
- **Transfer overhead**: Use pinned memory and consider data residency for iterative workloads

## Limitations and Future Work

### Current Limitations
- Host-side timing with `cudaDeviceSynchronize()` (sufficient for comparisons, not cycle-accurate)
- PCIe transfer overhead dominates small batches
- CUDA streams for overlapping H2D/D2H with compute is not a good option because the kernel is too fast, and there 
is an additional overhead for using the streams which are greater than the kernel time

### Future Enhancements
- **CUDA Events**: Device-side kernel timing for finer granularity
- **CUDA Graphs**: Reduce launch overhead for small batches
- **cuBLAS baseline**: Strided batched SGEMM for comparison
- **Tensor Cores**: Explore WMMA/CUTLASS for 4×4 blocks
- **CPU SIMD**: Explicit vectorization (AVX2/AVX-512/NEON)
- **Memory layouts**: Structure-of-arrays (SoA) experimentation

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `nvidia-smi: command not found` | Ensure NVIDIA driver installed; manually set `SM` (see Manual Override) |
| `nvcc fatal: Unsupported gpu architecture` | Choose supported SM or upgrade CUDA Toolkit |
| OpenMP not enabled | Verify `-fopenmp` compiler support |
| Verification mismatches | Check driver/toolkit versions; try tolerance 1e-3 |
| Out-of-memory | Reduce batch size; GPU must fit 4 input + 1 output batch |

## Citation

If you use this work, please cite:
```
ECE455 Final Project: Performance Analysis of Generic Approaches for Large-Scale Small-Size Matrix Multiplication
University of Wisconsin - Madison, Fall 2025
```

## License

This project is provided for educational purposes as part of ECE455 coursework.
