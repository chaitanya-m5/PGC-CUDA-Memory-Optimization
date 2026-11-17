# Experiment 6 – Memory Optimization in CUDA

### Coalesced vs. Uncoalesced Memory Access | Matrix Transpose | Naive Matrix Multiplication

![CUDA](https://img.shields.io/badge/CUDA-13.0-green?logo=nvidia)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20Tesla%20T4-green?logo=nvidia)
![Language](https://img.shields.io/badge/Language-CUDA%20C%2B%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange?logo=googlecolab)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 1. Introduction

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform that enables developers to execute computationally intensive operations on Graphics Processing Units (GPUs).

GPU performance depends not only on the number of processing cores but also on how efficiently threads access memory.

One of the important techniques for improving GPU memory performance is **memory coalescing**.

Memory coalescing allows the GPU to combine memory requests from threads in a warp into fewer, wider memory transactions. This improves effective global-memory bandwidth and reduces unnecessary memory traffic.

This experiment explores CUDA memory optimization through matrix transpose operations and studies the performance characteristics of naive matrix multiplication.

## 2. Aim

To study how the global-memory access pattern of a CUDA kernel affects its performance by:

1. Comparing an uncoalesced matrix transpose with a coalesced matrix transpose.
2. Implementing shared-memory tiling to optimize memory access.
3. Running naive matrix multiplication on the GPU.
4. Measuring kernel execution times using CUDA events.
5. Calculating effective memory bandwidth and GFLOPS.
6. Verifying GPU computation results against CPU reference calculations.

## 3. Objectives

- Understand global-memory access patterns in CUDA.
- Study coalesced and uncoalesced memory transactions.
- Understand the CUDA warp execution model.
- Implement two-dimensional CUDA kernels using `dim3` grids and blocks.
- Use shared memory to optimize matrix transpose.
- Understand shared-memory bank conflicts and padding.
- Measure GPU execution time using CUDA events.
- Calculate memory bandwidth and floating-point performance.
- Compare the performance of different CUDA implementations.

## 4. Hardware and Software Requirements

### Hardware Configuration

| Component | Specification |
|---|---|
| GPU | NVIDIA Tesla T4 |
| GPU Memory | 15 GB |
| GPU Architecture | NVIDIA Turing |
| Compute Capability | 7.5 |
| Execution Platform | Google Colab |

### Software Configuration

| Component | Version / Technology |
|---|---|
| Programming Language | CUDA C/C++ |
| CUDA Toolkit | 13.0 |
| CUDA Compiler | NVCC 13.0.88 |
| Development Environment | Google Colab |
| Operating Environment | Linux |
| GPU Monitoring | NVIDIA-SMI |

**Note:** This experiment was configured and executed using Google Colab's NVIDIA Tesla T4 GPU runtime. Performance measurements are hardware-dependent and may differ on other GPUs.

## 5. CUDA Memory Hierarchy

CUDA provides different types of memory that serve different purposes.

### 5.1 Global Memory

- Located in GPU device memory.
- Accessible by threads across GPU blocks.
- Used to store large matrices and arrays.
- Has higher latency than on-chip memory.
- Efficient access patterns are important for achieving good performance.

### 5.2 Shared Memory

- Located on the GPU multiprocessor.
- Shared by threads belonging to the same thread block.
- Provides faster access than global memory.
- Useful for data reuse and reorganizing memory access patterns.
- Requires synchronization when threads share data.

### 5.3 Registers

- Private to individual CUDA threads.
- Used to store temporary variables and intermediate calculations.
- Provide very fast access.

### 5.4 Memory Coalescing

A CUDA warp consists of 32 threads.

When neighbouring threads access neighbouring global-memory addresses, the hardware can combine their requests into fewer memory transactions.

This is called **coalesced memory access**.

When threads access addresses separated by large strides, more memory transactions may be required. This is called **uncoalesced memory access**.

| Feature | Coalesced Access | Uncoalesced Access |
|---|---|---|
| Address pattern | Consecutive or neighbouring | Scattered or strided |
| Memory transactions | More efficient | Less efficient |
| Bandwidth utilization | Generally higher | Generally lower |
| Performance | Generally better | Can be slower |

## 6. Experiments Implemented

### Experiment 1: Uncoalesced Matrix Transpose

Matrix transpose converts rows into columns.

For an input matrix A, the output matrix B is:

\[
B[j][i] = A[i][j]
\]

In row-major storage, an element at row r and column c is stored at:

\[
index = r \times N + c
\]

The uncoalesced transpose assigns one matrix element to each CUDA thread.

**Implementation characteristics:**

- Each CUDA thread copies one matrix element.
- Consecutive threads read consecutive elements from a row.
- Output writes are separated by the matrix dimension.
- Global-memory writes are strided and inefficient.

The main CUDA operation is:

```cpp
out[x * N + y] = in[y * N + x];
```

**Limitation:**

Although the input reads are coalesced, the output writes are separated by a stride of N. This leads to inefficient global-memory transactions.

### Experiment 2: Coalesced Matrix Transpose

The coalesced transpose improves memory access using shared-memory tiling.

Instead of directly transposing elements in global memory, the kernel temporarily stores a matrix tile in shared memory.

**Working procedure:**

1. Load a 32 × 32 tile from global memory into shared memory.
2. Use coalesced global-memory reads.
3. Synchronize threads using `__syncthreads()`.
4. Exchange row and column coordinates.
5. Read the tile in transposed order.
6. Write the transposed tile to global memory using coalesced writes.

**Shared-memory declaration:**

```cpp
__shared__ float tile[32][33];
```

The extra column provides padding to help avoid shared-memory bank conflicts.

**Thread synchronization:**

```cpp
__syncthreads();
```

This ensures that all threads complete loading the shared-memory tile before any thread reads its transposed values.

**Advantages:**

- Improves global-memory access patterns.
- Reduces inefficient memory transactions.
- Makes better use of available memory bandwidth.
- Demonstrates shared-memory tiling.
- Helps avoid shared-memory bank conflicts through padding.

### Experiment 3: Naive Matrix Multiplication

Matrix multiplication calculates the product of two square matrices.

For matrices A and B:

\[
C[i][j] = \sum_{k=0}^{N-1} A[i][k] \times B[k][j]
\]

In this implementation, each CUDA thread computes one output element of matrix C.

**Implementation characteristics:**

- Uses a two-dimensional CUDA grid.
- Uses 16 × 16 threads per block.
- Each thread calculates one output matrix element.
- Uses a loop of N iterations.
- Reads input values directly from global memory.
- Does not use shared-memory tiling for input data reuse.

The main CUDA calculation is:

```cpp
float sum = 0.0f;

for (int k = 0; k < N; k++)
{
    sum += A[row * N + k] * B[k * N + col];
}

C[row * N + col] = sum;
```

**Limitation:**

The naive implementation repeatedly accesses input data from global memory. It does not reuse matrix tiles through shared memory, which limits memory efficiency.

Shared-memory tiled matrix multiplication is a possible future optimization.

## 7. CUDA Kernel Configuration

| Parameter | Value |
|---|---:|
| Transpose tile dimension | 32 |
| Transpose block dimensions | 32 × 32 |
| Matrix multiplication block dimensions | 16 × 16 |
| Shared-memory tile | 32 × 33 |
| Transpose repetitions | 20 |
| Matrix multiplication repetitions | 5 |

The transpose kernels use 32 × 32 blocks, while matrix multiplication uses 16 × 16 blocks.

## 8. Execution-Time Measurement

CUDA events are used to measure GPU kernel execution time.

The measurement procedure is:

1. Create CUDA start and stop events.
2. Perform a warm-up kernel launch.
3. Record the start event.
4. Execute the kernel repeatedly.
5. Record the stop event.
6. Synchronize the GPU.
7. Calculate the elapsed time.
8. Divide the total time by the number of repetitions.

Average execution time:

\[
T_{avg} = \frac{T_{total}}{R}
\]

where R is the number of repetitions.

The experiment uses 20 repetitions for transpose kernels and 5 repetitions for naive matrix multiplication.

Warm-up executions are excluded from timing.

## 9. Performance Metrics

### 9.1 Effective Memory Bandwidth

Memory bandwidth measures how much data is transferred per unit of time.

For matrix transpose:

\[
Bandwidth = \frac{2 \times N^2 \times 4}{T}
\]

where:

- N = matrix dimension.
- 4 = bytes per float.
- T = execution time in seconds.
- Factor 2 = one matrix read and one matrix write.

The reported bandwidth is in GB/s.

### 9.2 Speedup

Speedup measures how much faster the coalesced transpose is compared with the uncoalesced transpose.

\[
Speedup = \frac{T_{uncoalesced}}{T_{coalesced}}
\]

Interpretation:

- Greater than 1: coalesced transpose is faster.
- Equal to 1: both have equal execution time.
- Less than 1: coalesced transpose is slower for that measurement.

### 9.3 GFLOPS

GFLOPS measures billions of floating-point operations per second.

For naive matrix multiplication:

\[
GFLOPS = \frac{2 \times N^3}{T \times 10^9}
\]

where T is measured in seconds.

The factor of 2 represents one multiplication and one addition for each iteration.

## 10. Default Matrix Sizes

| Operation | Matrix Size |
|---|---|
| Uncoalesced Transpose | 4096 × 4096 |
| Coalesced Transpose | 4096 × 4096 |
| Naive Matrix Multiplication | 1024 × 1024 |

Custom matrix dimensions can also be supplied through command-line arguments.

For example:

```bash
./exp6 2048 512
```

The first argument sets the transpose matrix dimension, and the second argument sets the matrix multiplication dimension.

## 11. Compilation and Execution

### Step 1: Verify GPU availability

```bash
nvidia-smi
```

### Step 2: Verify CUDA compiler

```bash
nvcc --version
```

### Step 3: Compile the CUDA program

For NVIDIA Tesla T4 with compute capability 7.5:

```bash
nvcc -O2 -arch=sm_75 -o exp6 exp6.cu
```

### Step 4: Run the experiment

```bash
./exp6
```

### Step 5: Run with custom dimensions

```bash
./exp6 2048 512
```

### Step 6: Save execution output

```bash
./exp6 | tee results.txt
```

## 12. Result Verification

The program verifies the correctness of its GPU computations.

| Kernel | Verification Method |
|---|---|
| Uncoalesced Transpose | CPU comparison of transposed elements |
| Coalesced Transpose | CPU comparison of transposed elements |
| Naive Matrix Multiplication | CPU reference calculation on 20 sampled output elements |

A successful execution should report PASS for all three kernels.

The complete execution output is stored in `results.txt`.

## 13. Performance Results and Graphical Analysis

The actual performance values measured on the NVIDIA Tesla T4 GPU are recorded in `results.txt` and `results.csv`.

### Graph 1: Matrix Transpose Execution-Time Comparison

This graph compares the average execution time of the uncoalesced and coalesced transpose kernels.

**Expected interpretation:** A shorter bar indicates a faster execution. The coalesced transpose is expected to perform better because it uses shared memory to achieve coalesced global-memory reads and writes.

![Transpose Execution Time](screenshots/transpose_time_comparison.png)

### Graph 2: Effective Memory Bandwidth Comparison

This graph compares the effective memory bandwidth of both transpose implementations.

![Transpose Memory Bandwidth](screenshots/transpose_bandwidth_comparison.png)

**Expected interpretation:** Higher bandwidth indicates more efficient global-memory data transfer.

### Graph 3: Naive Matrix Multiplication Performance

This graph displays the measured floating-point throughput of the naive matrix multiplication kernel.

![Naive Matrix Multiplication GFLOPS](screenshots/matmul_gflops.png)

**Expected interpretation:** The GFLOPS value indicates the floating-point processing rate achieved by the naive matrix multiplication implementation on the selected GPU.

**Important:** These graph images are generated from actual measurements using the plotting script. Do not substitute illustrative or estimated performance values.

## 14. Observation Table

Record at least three executions as required by the lab manual.

| Run | Uncoalesced Time (ms) | Coalesced Time (ms) | Speedup |
|---|---:|---:|---:|
| 1 | Fill from measurement | Fill from measurement | Calculate |
| 2 | Fill from measurement | Fill from measurement | Calculate |
| 3 | Fill from measurement | Fill from measurement | Calculate |

### Matrix Multiplication

| Parameter | Value |
|---|---|
| Matrix dimension | 1024 × 1024 |
| Average execution time | Fill from measurement |
| GFLOPS | Fill from measurement |
| Verification | PASS / FAIL |

GPU model: NVIDIA Tesla T4  
Compute capability: 7.5

## 15. Analysis and Discussion

### Why does coalesced transpose perform better?

In the uncoalesced transpose, neighbouring threads write to memory locations separated by N elements. This creates inefficient memory transactions.

The coalesced transpose first loads a tile into shared memory and then writes the transposed values to global memory in a coalesced pattern.

### Why is shared-memory padding used?

The shared-memory tile is declared as `[32][33]` rather than `[32][32]`.

The additional column helps avoid shared-memory bank conflicts when threads access the tile in transposed order.

### Why is `__syncthreads()` necessary?

All threads must finish loading the shared-memory tile before other threads read its transposed contents.

`__syncthreads()` provides synchronization within a thread block.

### Why is naive matrix multiplication less efficient?

Each output thread repeatedly reads values from the input matrices through global memory.

The kernel does not reuse input tiles through shared memory, causing repeated memory traffic.

### Why use CUDA events?

CUDA kernel launches are asynchronous. CUDA events measure execution time on the GPU and help avoid inaccurate CPU-only timing around asynchronous launches.

## 16. Applications

CUDA memory optimization is useful in:

- Scientific computing.
- Image processing.
- Computer vision.
- Deep learning.
- Neural networks.
- Numerical simulations.
- Large-scale matrix operations.
- High-performance computing.
- GPU-accelerated data processing.

## 17. Future Enhancements

The experiment can be extended through:

1. Shared-memory tiled matrix multiplication.
2. Comparison of different matrix tile dimensions such as 8, 16 and 32.
3. Testing larger matrix dimensions.
4. Comparison across different NVIDIA GPU architectures.
5. Removing shared-memory padding and measuring the effect of bank conflicts.
6. Comparing memory bandwidth across multiple GPU configurations.

## 18. Conclusion

This experiment demonstrates that GPU performance depends significantly on memory access patterns.

The uncoalesced transpose shows the performance limitations of strided global-memory writes. The coalesced transpose demonstrates how shared-memory tiling, synchronization and padding can improve memory access efficiency.

The naive matrix multiplication kernel demonstrates parallel computation on the GPU while highlighting the performance limitations caused by repeated global-memory access without shared-memory data reuse.

Overall, the experiment provides practical understanding of CUDA memory hierarchy, memory coalescing, shared memory, bank conflicts, CUDA event timing, memory bandwidth and GPU performance evaluation.

## 19. Repository Structure

```text
CUDA-Experiment-6-Memory-Optimization/
│
├── exp6.cu
├── README.md
├── results.txt
├── results.csv
├── plot_results.py
│
└── screenshots/
    ├── 01-colab-gpu-and-cuda-toolkit-setup.png
    ├── 02-cuda-program-compilation.png
    ├── 03-matrix-transpose-execution-results.png
    ├── 04-matrix-multiplication-results.png
    ├── transpose_time_comparison.png
    ├── transpose_bandwidth_comparison.png
    └── matmul_gflops.png
```

## 20. References

1. NVIDIA CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
2. NVIDIA CUDA Toolkit Documentation: https://docs.nvidia.com/cuda/
3. NVIDIA CUDA C++ Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

---

**Experiment 6 – Memory Optimization in CUDA**  
*CUDA C/C++ | NVIDIA GPU | Shared Memory | Global Memory | Matrix Transpose | Matrix Multiplication*
