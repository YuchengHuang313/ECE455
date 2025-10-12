# Portable CUDA Matrix Multiplication Benchmark

Auto-detecting CUDA build system for batched 4×4 matrix multiplication.

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
