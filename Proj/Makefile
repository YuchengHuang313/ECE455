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
TARGET_MEM_LAYOUT = compare_mem_layout
TARGET_VAR_JOINTS = compare_variable_joints
TARGET_CORRECTNESS = test_combined_correctness
TARGET_MEM_ACCESS = compare_mem_access
CU_SOURCES = small_matmul.cu compare_mem_layout.cu compare_variable_joints.cu test_combined_correctness.cu compare_mem_access.cu
HEADERS = small_matmul.cuh
OBJECTS_MEM_LAYOUT = small_matmul.o compare_mem_layout.o
OBJECTS_VAR_JOINTS = small_matmul.o compare_variable_joints.o
OBJECTS_CORRECTNESS = small_matmul.o test_combined_correctness.o

# Default
all: $(TARGET_MEM_LAYOUT) $(TARGET_VAR_JOINTS) $(TARGET_CORRECTNESS) $(TARGET_MEM_ACCESS)

$(TARGET_MEM_LAYOUT): $(OBJECTS_MEM_LAYOUT)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_VAR_JOINTS): $(OBJECTS_VAR_JOINTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_CORRECTNESS): $(OBJECTS_CORRECTNESS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

small_matmul.o: small_matmul.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

compare_mem_layout.o: compare_mem_layout.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

compare_variable_joints.o: compare_variable_joints.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

test_combined_correctness.o: test_combined_correctness.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(TARGET_MEM_ACCESS): compare_mem_access.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

compare_mem_access.o: compare_mem_access.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

run: $(TARGET_VAR_JOINTS)
	./$(TARGET_VAR_JOINTS) 1000000

run-mem-layout: $(TARGET_MEM_LAYOUT)
	./$(TARGET_MEM_LAYOUT) 1000000

run-mem-layout-small: $(TARGET_MEM_LAYOUT)
	./$(TARGET_MEM_LAYOUT) 10000

run-mem-layout-large: $(TARGET_MEM_LAYOUT)
	./$(TARGET_MEM_LAYOUT) 5000000

run-joints: $(TARGET_VAR_JOINTS)
	./$(TARGET_VAR_JOINTS) 1000000

run-joints-small: $(TARGET_VAR_JOINTS)
	./$(TARGET_VAR_JOINTS) 10000

run-joints-large: $(TARGET_VAR_JOINTS)
	./$(TARGET_VAR_JOINTS) 5000000

run-correctness: $(TARGET_CORRECTNESS)
	./$(TARGET_CORRECTNESS)

run-mem-access: $(TARGET_MEM_ACCESS)
	./$(TARGET_MEM_ACCESS)

run-mem-access-large: $(TARGET_MEM_ACCESS)
	./$(TARGET_MEM_ACCESS) 512

clean:
	rm -f *.o $(TARGET_MEM_LAYOUT) $(TARGET_VAR_JOINTS) $(TARGET_CORRECTNESS) $(TARGET_MEM_ACCESS)

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

.PHONY: all run run-mem-layout run-mem-layout-small run-mem-layout-large run-joints run-joints-small run-joints-large run-correctness clean rebuild gpuinfo info
