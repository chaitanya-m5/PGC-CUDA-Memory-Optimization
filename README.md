# Experiment 6: Memory Optimization in CUDA

## Aim
To study CUDA memory optimization techniques by comparing uncoalesced and coalesced matrix transpose and implementing naive matrix multiplication.

## Hardware and Software Environment
- GPU: NVIDIA Tesla T4
- GPU Memory: 15 GB
- CUDA Toolkit: 13.0
- Language: CUDA C++

## Experiments Performed

### 1. Uncoalesced Matrix Transpose
A basic matrix transpose kernel where input reads are coalesced but output writes are strided, resulting in inefficient global memory access.

### 2. Coalesced Matrix Transpose
Uses shared-memory tiling to improve global memory access.
- Shared-memory tile: 32 x 33
- Synchronization using `__syncthreads()`
- Padding to reduce shared-memory bank conflicts
- Improved memory coalescing

### 3. Naive Matrix Multiplication
Each CUDA thread computes one output element using the naive matrix multiplication algorithm.

## Implementation Details
- CUDA kernel implementations
- CUDA event-based timing
- Warm-up execution
- Repeated kernel execution
- CPU-side result verification

## Default Matrix Sizes

| Operation | Matrix Size |
|---|---|
| Matrix Transpose | 4096 x 4096 |
| Matrix Multiplication | 1024 x 1024 |

## Compilation

For NVIDIA Tesla T4:

```bash
nvcc -O2 -arch=sm_75 -o exp6 exp6.cu
```

## Execution

```bash
./exp6
```

For custom matrix sizes:

```bash
./exp6 2048 512
```

## Performance Results

See `results.txt` for the actual execution times, memory bandwidth, GFLOPS and correctness verification.

## Files

- `exp6.cu` - CUDA source code
- `results.txt` - Experiment execution output
- `README.md` - Experiment documentation

## Conclusion

The experiment demonstrates the importance of coalesced global memory access and shared-memory tiling in improving CUDA memory efficiency. It also evaluates the performance of naive matrix multiplication.

---
CUDA Experiment 6 - Memory Optimization
