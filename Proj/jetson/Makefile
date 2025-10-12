# ===== Jetson Orin Nano Super (Ampere, SM 8.7) =====
NVCC ?= /usr/local/cuda/bin/nvcc
SM ?= 87
NVCC_FLAGS = -O3 -use_fast_math \
             -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -fopenmp \
             -gencode arch=compute_$(SM),code=sm_$(SM)
# Optional: show register/SMEM usage
# NVCC_FLAGS += -Xptxas=-v

LDFLAGS = -lgomp

# Target & sources
TARGET = small_matmul_test
CU_SOURCES  = small_matmul.cu
CPP_SOURCES = small_matmul.cpp
HEADERS     = small_matmul.cuh
OBJECTS     = small_matmul.o small_matmul_cpp.o

# Default
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

small_matmul.o: small_matmul.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

small_matmul_cpp.o: small_matmul.cpp $(HEADERS)
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
	-@/usr/local/cuda/bin/nvcc --version || true
	-@cat /etc/nv_tegra_release 2>/dev/null || true

.PHONY: all run run-small run-medium run-large clean rebuild gpuinfo
