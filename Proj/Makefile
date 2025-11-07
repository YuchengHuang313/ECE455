# ================================================================================
# Portable CUDA Makefile with Auto-detection
# ================================================================================
# This Makefile automatically detects:
#   1. CPU Architecture (x86_64 or aarch64/ARM)
#   2. GPU Compute Capability (SM) via nvidia-smi
#
# Usage:
#   make              - Auto-detect everything and build
#   make SM=86        - Override SM (e.g., for RTX 30xx)
#   make info         - Show detected system/GPU information
#   make run          - Run with default 1000 matrices
#   make run-large    - Run with 100,000 matrices
#
# Supported GPUs (SM versions):
#   SM 75: Turing (RTX 20xx, GTX 16xx)
#   SM 86: Ampere (RTX 30xx, A100)
#   SM 87: Ampere (Jetson Orin)
#   SM 89: Ada Lovelace (RTX 40xx)
# ================================================================================

# ===== Auto-detect Architecture and GPU Compute Capability =====
ARCH := $(shell uname -m)

# Try to auto-detect GPU compute capability if not specified
ifndef SM
    # Try using deviceQuery or nvidia-smi to detect GPU
    SM_DETECT := $(shell which nvidia-smi > /dev/null 2>&1 && \
                         nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | \
                         head -n1 | tr -d '.')
    
    # If detection failed, use architecture-based defaults
    ifeq ($(SM_DETECT),)
        ifeq ($(ARCH),x86_64)
            SM := 75
            $(info Could not detect GPU. Using default SM $(SM) for x86_64)
        else ifeq ($(ARCH),aarch64)
            SM := 87
            $(info Could not detect GPU. Using default SM $(SM) for Jetson)
        endif
    else
        SM := $(SM_DETECT)
        $(info Auto-detected GPU with SM $(SM))
    endif
endif

# Set NVCC path based on architecture
ifeq ($(ARCH),x86_64)
    NVCC ?= nvcc
    $(info Building for x86_64 with SM $(SM))
else ifeq ($(ARCH),aarch64)
    NVCC ?= /usr/local/cuda/bin/nvcc
    $(info Building for ARM (Jetson) with SM $(SM))
else
    $(error Unknown architecture: $(ARCH))
endif

NVCC_FLAGS = -O3 -use_fast_math -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -fopenmp
NVCC_FLAGS += -gencode arch=compute_$(SM),code=sm_$(SM)
# Optional: show register/SMEM usage
# NVCC_FLAGS += -Xptxas=-v

LDFLAGS = -lgomp

# Target & sources
TARGET = small_matmul_test
CU_SOURCES  = small_matmul.cu compare.cu
HEADERS     = small_matmul.cuh
OBJECTS     = small_matmul.o compare.o

# Default
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

small_matmul.o: small_matmul.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

compare.o: compare.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

run-small: $(TARGET)
	./$(TARGET) 100
run-medium: $(TARGET)
	./$(TARGET) 10000
run-large: $(TARGET)
	./$(TARGET) 100000

clean:
	rm -f $(OBJECTS) $(TARGET)

rebuild: clean all

gpuinfo:
	@echo "=== System Information ==="
	@echo "Architecture: $(ARCH)"
	@echo "Detected SM: $(SM)"
	@echo ""
	@echo "=== GPU Information ==="
	@nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv 2>/dev/null || echo "nvidia-smi not available"
	@echo ""
	@echo "=== CUDA Version ==="
	@nvcc --version 2>/dev/null || /usr/local/cuda/bin/nvcc --version 2>/dev/null || echo "nvcc not found"
	@echo ""
	@echo "=== Jetson Info (if applicable) ==="
	@cat /etc/nv_tegra_release 2>/dev/null || echo "Not a Jetson device"

info: gpuinfo

.PHONY: all run run-small run-medium run-large clean rebuild gpuinfo info
