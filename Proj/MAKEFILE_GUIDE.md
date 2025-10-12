# Complete Makefile Guide for Beginners

## What is a Makefile?
A Makefile is a script that automates compilation. Instead of typing long compiler commands, you just type `make` and it handles everything.

---

## Basic Makefile Syntax

### Variables
```makefile
VARIABLE_NAME = value
```
- Use variables like: `$(VARIABLE_NAME)`
- Like variables in programming languages

### Comments
```makefile
# This is a comment
```

### Rules (The Core Concept)
```makefile
target: dependencies
	command
```
- **target**: What you want to build (e.g., a `.o` file or executable)
- **dependencies**: What files are needed (if they change, rebuild)
- **command**: How to build it (MUST start with a TAB, not spaces!)

---

## Line-by-Line Explanation of Your Makefile

### 1. Auto-Detect CPU Architecture
```makefile
ARCH := $(shell uname -m)
```
**What it does:**
- `:=` means "assign immediately" (evaluated once)
- `$(shell ...)` runs a shell command
- `uname -m` prints CPU architecture:
  - `x86_64` = Intel/AMD computers
  - `aarch64` = ARM processors (Jetson, Mac M1/M2)

**Why:** Different systems need different CUDA paths.

---

### 2. Auto-Detect GPU Compute Capability
```makefile
ifndef SM
    SM_DETECT := $(shell which nvidia-smi > /dev/null 2>&1 && \
                         nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | \
                         head -n1 | tr -d '.')
```

**Breaking it down:**

**`ifndef SM`** = "if SM variable is NOT defined"
- Allows manual override: `make SM=86`

**`which nvidia-smi > /dev/null 2>&1`**
- `which` finds if command exists
- `> /dev/null` throws away output
- `2>&1` redirects errors to same place
- Returns success if found

**`&&`** = "and then" (only runs next command if first succeeded)

**`nvidia-smi --query-gpu=compute_cap --format=csv,noheader`**
- Asks GPU for compute capability
- Returns: `8.9` (for RTX 40 series)

**`head -n1`** = take first line (if multiple GPUs)

**`tr -d '.'`** = delete the dot
- Transforms `8.9` → `89`

---

### 3. Fallback to Defaults
```makefile
ifeq ($(SM_DETECT),)
    ifeq ($(ARCH),x86_64)
        SM := 75
```

**What it does:**
- If detection failed (empty result), use sensible defaults
- `ifeq` = "if equal"
- x86 → default SM 75 (RTX 20 series)
- ARM → default SM 87 (Jetson Orin)

---

### 4. Set Compiler Path
```makefile
ifeq ($(ARCH),x86_64)
    NVCC ?= nvcc
```

**`?=`** = "assign if not already set"
- Allows override: `NVCC=/custom/path/nvcc make`

**Why different paths?**
- x86: `nvcc` (assumes CUDA in system PATH)
- Jetson: `/usr/local/cuda/bin/nvcc` (specific location)

---

### 5. Compiler Flags
```makefile
NVCC_FLAGS = -O3 -use_fast_math -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -fopenmp
NVCC_FLAGS += -gencode arch=compute_$(SM),code=sm_$(SM)
```

**Breaking down each flag:**

**`-O3`** = Optimization level 3 (maximum speed)

**`-use_fast_math`** = Fast but less precise math (OK for many apps)

**`-Xcompiler`** = Pass next flag to C++ compiler
- `-Wall` = Show all warnings
- `-Wextra` = Show extra warnings
- `-fopenmp` = Enable OpenMP (multithreading)

**`+=`** = Append to variable

**`-gencode arch=compute_$(SM),code=sm_$(SM)`**
- Tells CUDA which GPU architecture to compile for
- `compute_89` = Virtual architecture (intermediate code)
- `sm_89` = Real architecture (specific GPU)
- Using `$(SM)` makes it flexible

---

### 6. Linker Flags
```makefile
LDFLAGS = -lgomp
```

**`-lgomp`** = Link OpenMP library
- `l` means "library"
- `gomp` = GNU OpenMP implementation

---

### 7. Define Files
```makefile
TARGET = small_matmul_test          # Final executable name
CU_SOURCES  = small_matmul.cu       # CUDA source file
CPP_SOURCES = small_matmul.cpp      # C++ source file
HEADERS     = small_matmul.cuh      # Header file
OBJECTS     = small_matmul.o small_matmul_cpp.o  # Compiled object files
```

**Why object files (.o)?**
- Source → Object → Executable
- Only recompile changed files (faster)

---

### 8. Default Target
```makefile
all: $(TARGET)
```

**What happens when you type `make`:**
1. Make looks for first rule (this one)
2. `all` depends on `$(TARGET)` (small_matmul_test)
3. Make checks if `small_matmul_test` exists and is up-to-date
4. If not, builds it using the `$(TARGET)` rule below

---

### 9. Build Executable
```makefile
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)
```

**Translation:**
"To build `small_matmul_test`, I need `small_matmul.o` and `small_matmul_cpp.o`"

**Special variables:**
- `$@` = The target name (`small_matmul_test`)
- `$^` = All dependencies (`small_matmul.o small_matmul_cpp.o`)

**Actual command:**
```bash
nvcc -O3 ... -o small_matmul_test small_matmul.o small_matmul_cpp.o -lgomp
```

---

### 10. Build Object Files
```makefile
small_matmul.o: small_matmul.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
```

**Translation:**
"To build `small_matmul.o`, I need `small_matmul.cu` and `small_matmul.cuh`"

**`-c`** = Compile only (don't link)

**`$<`** = First dependency (`small_matmul.cu`)

**Flow:**
```
small_matmul.cu → [compile] → small_matmul.o
```

---

### 11. Convenience Commands
```makefile
run: $(TARGET)
	./$(TARGET)
```

**What it does:**
- `make run` builds (if needed) then runs program

**Dependency chain:**
```
run → $(TARGET) → $(OBJECTS) → source files
```

If any source changed, rebuilds automatically!

---

### 12. Test Commands
```makefile
run-small: $(TARGET)
	./$(TARGET) 100
```

**Usage:**
- `make run-small` → Runs with 100 matrices
- `make run-medium` → Runs with 10,000 matrices
- `make run-large` → Runs with 100,000 matrices

---

### 13. Clean Up
```makefile
clean:
	rm -f $(OBJECTS) $(TARGET)
```

**`rm -f`** = Remove files, don't error if missing

**Removes:**
- `small_matmul.o`
- `small_matmul_cpp.o`
- `small_matmul_test`

**Usage:** `make clean` before rebuilding from scratch

---

### 14. Rebuild Everything
```makefile
rebuild: clean all
```

**Order matters!**
1. Run `clean` (delete old files)
2. Run `all` (build fresh)

**Usage:** `make rebuild`

---

### 15. Info Commands
```makefile
gpuinfo:
	@echo "=== System Information ==="
	@echo "Architecture: $(ARCH)"
```

**`@`** = Don't print the command itself
- Without `@`: prints `echo "text"` then `text`
- With `@`: only prints `text`

**`||`** = "or" (if first fails, try second)
```makefile
@nvcc --version 2>/dev/null || echo "nvcc not found"
```

---

### 16. Phony Targets
```makefile
.PHONY: all run run-small run-medium run-large clean rebuild gpuinfo info
```

**What's a phony target?**
- Not real files (just command names)
- Without `.PHONY`: if you had a file named `clean`, `make clean` wouldn't work

---

## Common Make Commands

```bash
make              # Build (auto-detect everything)
make clean        # Delete compiled files
make rebuild      # Clean then build
make run          # Build and run
make info         # Show system/GPU info
make SM=86        # Build for specific GPU
```

---

## Make's Smart Features

### 1. Incremental Builds
```
Edit small_matmul.cu → make
  ↓
Only recompiles small_matmul.o (not small_matmul_cpp.o)
  ↓
Relinks executable
```

### 2. Dependency Tracking
```makefile
small_matmul.o: small_matmul.cu small_matmul.cuh
```
If you edit the header, make knows to rebuild!

### 3. Parallel Builds
```bash
make -j8  # Use 8 parallel jobs (faster on multi-core)
```

---

## How Make Decides What to Build

Make compares **timestamps**:

```
small_matmul.cu (modified: 10:00 AM)
small_matmul.o  (created:  9:00 AM)
```

**Result:** Source is newer → rebuild object file

---

## Common Pitfalls

### 1. Tabs vs Spaces
```makefile
target: deps
    command     # ✗ WRONG (spaces)
	command     # ✓ CORRECT (tab)
```

### 2. Forgetting Dependencies
```makefile
program: main.o
	gcc -o program main.o

# ✗ Missing: what does main.o depend on?
```

Should be:
```makefile
main.o: main.c header.h
	gcc -c main.c
```

---

## Your Makefile Flow Diagram

```
Type: make
  ↓
Check: all (depends on small_matmul_test)
  ↓
Check: small_matmul_test (depends on .o files)
  ↓
Check: small_matmul.o (depends on .cu, .cuh)
  ├─ If .cu/.cuh changed → compile
  ↓
Check: small_matmul_cpp.o (depends on .cpp, .cuh)
  ├─ If .cpp/.cuh changed → compile
  ↓
Link: small_matmul.o + small_matmul_cpp.o → small_matmul_test
  ↓
Done!
```

---

## Quick Reference

| Syntax | Meaning |
|--------|---------|
| `VAR = value` | Define variable |
| `$(VAR)` | Use variable |
| `target: deps` | Build rule |
| `$@` | Target name |
| `$<` | First dependency |
| `$^` | All dependencies |
| `:=` | Immediate assignment |
| `?=` | Assign if not set |
| `+=` | Append |
| `@command` | Don't echo command |
| `.PHONY:` | Not a real file |
| `ifeq (x,y)` | If equal |
| `ifndef VAR` | If not defined |

---

## Practice Exercises

1. **Add a debug build:**
   ```makefile
   debug: NVCC_FLAGS += -g -G
   debug: all
   ```
   Usage: `make debug`

2. **Add verbose mode:**
   ```makefile
   verbose: NVCC_FLAGS += -Xptxas=-v
   verbose: all
   ```

3. **Count lines of code:**
   ```makefile
   count:
   	@wc -l $(CU_SOURCES) $(CPP_SOURCES) $(HEADERS)
   ```

Try modifying the Makefile yourself! 🚀
