# Compiler settings
NVCC = nvcc
CUDA_ARCH = sm_75  # Change this based on your GPU (sm_75 for Turing, sm_86 for Ampere, etc.)

# Compiler flags
NVCC_FLAGS = -arch=$(CUDA_ARCH) -O3 -use_fast_math
NVCC_FLAGS += -Xcompiler -Wall -Xcompiler -Wextra
NVCC_FLAGS += -Xcompiler -fopenmp
LDFLAGS = -lgomp

# Target executable
TARGET = small_matmul_test

# Source files
CU_SOURCES = small_matmul.cu
CPP_SOURCES = small_matmul.cpp
HEADERS = small_matmul.cuh

# Object files
OBJECTS = small_matmul.o small_matmul_cpp.o

# Default target
all: $(TARGET)

# Link the final executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

# Compile CUDA source
small_matmul.o: small_matmul.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source with CUDA
small_matmul_cpp.o: small_matmul.cpp $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Run the test
run: $(TARGET)
	./$(TARGET)

# Run with custom number of matrices
run-small: $(TARGET)
	./$(TARGET) 100

run-medium: $(TARGET)
	./$(TARGET) 10000

run-large: $(TARGET)
	./$(TARGET) 100000

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Rebuild everything
rebuild: clean all

# Show GPU info
gpuinfo:
	nvidia-smi

.PHONY: all run run-small run-medium run-large clean rebuild gpuinfo
