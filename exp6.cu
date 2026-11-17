
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32
#define BLK 16

// ---------- CUDA ERROR CHECKING MACRO ----------
#define CHECK(call) do { \
    cudaError_t _cuda_status = (call); \
    if (_cuda_status != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                cudaGetErrorString(_cuda_status), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


// ============================================================
// KERNEL 1: UNCOALESCED MATRIX TRANSPOSE
// ============================================================

__global__ void transposeNaive(const float *in, float *out, int N)
{
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < N && y < N)
    {
        out[x * N + y] = in[y * N + x];
    }
}


// ============================================================
// KERNEL 2: COALESCED MATRIX TRANSPOSE
// ============================================================

__global__ void transposeCoalesced(const float *in, float *out, int N)
{
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // Read input matrix into shared memory
    if (x < N && y < N)
    {
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    }

    __syncthreads();

    // Swap block coordinates for transposed output
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    // Write transposed data to output matrix
    if (x < N && y < N)
    {
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}


// ============================================================
// KERNEL 3: NAIVE MATRIX MULTIPLICATION
// ============================================================

__global__ void matMulNaive(const float *A, const float *B,
                            float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}


// ============================================================
// HOST HELPER: TRANSPOSE TIMING
// ============================================================

float timeTranspose(bool useCoalesced,
                    dim3 grid, dim3 block,
                    const float *in, float *out,
                    int N, int reps)
{
    cudaEvent_t startEvent, stopEvent;

    CHECK(cudaEventCreate(&startEvent));
    CHECK(cudaEventCreate(&stopEvent));

    CHECK(cudaMemset(out, 0, (size_t)N * N * sizeof(float)));

    // Warm-up launch (not timed)
    if (useCoalesced)
        transposeCoalesced<<<grid, block>>>(in, out, N);
    else
        transposeNaive<<<grid, block>>>(in, out, N);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Start timing
    CHECK(cudaEventRecord(startEvent));

    // Repeated kernel executions
    for (int r = 0; r < reps; r++)
    {
        if (useCoalesced)
            transposeCoalesced<<<grid, block>>>(in, out, N);
        else
            transposeNaive<<<grid, block>>>(in, out, N);
    }

    // Stop timing
    CHECK(cudaEventRecord(stopEvent));
    CHECK(cudaEventSynchronize(stopEvent));

    float ms = 0.0f;

    CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));

    CHECK(cudaEventDestroy(startEvent));
    CHECK(cudaEventDestroy(stopEvent));

    return ms / reps;
}


// ============================================================
// HOST HELPER: VERIFY MATRIX TRANSPOSE
// ============================================================

bool checkTranspose(const float *in, const float *out, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (out[j * N + i] != in[i * N + j])
                return false;
        }
    }

    return true;
}


// ============================================================
// HOST HELPER: VERIFY MATRIX MULTIPLICATION
// ============================================================

bool checkMatMul(const float *A, const float *B,
                 const float *C, int N)
{
    for (int t = 0; t < 20; t++)
    {
        int i = rand() % N;
        int j = rand() % N;

        double ref = 0.0;

        for (int k = 0; k < N; k++)
        {
            ref += (double)A[i * N + k] * B[k * N + j];
        }

        if (fabs(ref - C[i * N + j]) > 1e-3 * fabs(ref) + 1e-3)
            return false;
    }

    return true;
}


// ============================================================
// MAIN FUNCTION
// ============================================================

int main(int argc, char **argv)
{
    int NT = 4096;
    int NM = 1024;

    if (argc > 1) NT = atoi(argv[1]);
    if (argc > 2) NM = atoi(argv[2]);

    if (NT <= 0 || NM <= 0)
    {
        printf("Error: Matrix sizes must be positive.\n");
        return EXIT_FAILURE;
    }

    // Detect GPU
    cudaDeviceProp prop;

    CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("============================================\n");
    printf(" EXPERIMENT 6: MEMORY OPTIMIZATION IN CUDA\n");
    printf("============================================\n");

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);


    // ========================================================
    // PART A: MATRIX TRANSPOSE
    // ========================================================

    size_t bytesT = (size_t)NT * NT * sizeof(float);

    float *h_in = (float *)malloc(bytesT);
    float *h_out = (float *)malloc(bytesT);

    if (h_in == NULL || h_out == NULL)
    {
        printf("Host memory allocation failed for transpose.\n");
        return EXIT_FAILURE;
    }

    // Initialize input matrix
    for (size_t i = 0; i < (size_t)NT * NT; i++)
    {
        h_in[i] = (float)(i % 1000);
    }

    float *d_in = NULL;
    float *d_out = NULL;

    CHECK(cudaMalloc(&d_in, bytesT));
    CHECK(cudaMalloc(&d_out, bytesT));

    CHECK(cudaMemcpy(d_in, h_in, bytesT,
                     cudaMemcpyHostToDevice));

    dim3 blockT(TILE, TILE);

    dim3 gridT((NT + TILE - 1) / TILE,
               (NT + TILE - 1) / TILE);

    // Run uncoalesced transpose
    float tN = timeTranspose(
        false, gridT, blockT,
        d_in, d_out, NT, 20);

    CHECK(cudaMemcpy(h_out, d_out, bytesT,
                     cudaMemcpyDeviceToHost));

    bool okN = checkTranspose(h_in, h_out, NT);

    // Run coalesced transpose
    float tC = timeTranspose(
        true, gridT, blockT,
        d_in, d_out, NT, 20);

    CHECK(cudaMemcpy(h_out, d_out, bytesT,
                     cudaMemcpyDeviceToHost));

    bool okC = checkTranspose(h_in, h_out, NT);

    double gbN = 2.0 * bytesT / (tN * 1e6);
    double gbC = 2.0 * bytesT / (tC * 1e6);

    printf("\n============================================\n");
    printf(" PART A: MATRIX TRANSPOSE (N = %d)\n", NT);
    printf("============================================\n");

    printf("Uncoalesced : %8.3f ms | %7.1f GB/s | %s\n",
           tN, gbN, okN ? "PASS" : "FAIL");

    printf("Coalesced   : %8.3f ms | %7.1f GB/s | %s\n",
           tC, gbC, okC ? "PASS" : "FAIL");

    printf("Speedup (uncoalesced / coalesced) = %.2fx\n",
           tN / tC);


    // ========================================================
    // PART B: NAIVE MATRIX MULTIPLICATION
    // ========================================================

    size_t bytesM = (size_t)NM * NM * sizeof(float);

    float *hA = (float *)malloc(bytesM);
    float *hB = (float *)malloc(bytesM);
    float *hC = (float *)malloc(bytesM);

    if (hA == NULL || hB == NULL || hC == NULL)
    {
        printf("Host memory allocation failed for matrix multiplication.\n");
        return EXIT_FAILURE;
    }

    // Initialize matrices
    for (size_t i = 0; i < (size_t)NM * NM; i++)
    {
        hA[i] = rand() / (float)RAND_MAX;
        hB[i] = rand() / (float)RAND_MAX;
    }

    float *dA = NULL;
    float *dB = NULL;
    float *dC = NULL;

    CHECK(cudaMalloc(&dA, bytesM));
    CHECK(cudaMalloc(&dB, bytesM));
    CHECK(cudaMalloc(&dC, bytesM));

    CHECK(cudaMemcpy(dA, hA, bytesM,
                     cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(dB, hB, bytesM,
                     cudaMemcpyHostToDevice));

    dim3 block(BLK, BLK);

    dim3 grid((NM + BLK - 1) / BLK,
              (NM + BLK - 1) / BLK);

    cudaEvent_t startEvent, stopEvent;

    CHECK(cudaEventCreate(&startEvent));
    CHECK(cudaEventCreate(&stopEvent));

    // Warm-up
    matMulNaive<<<grid, block>>>(dA, dB, dC, NM);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Measure five executions
    int reps = 5;

    CHECK(cudaEventRecord(startEvent));

    for (int r = 0; r < reps; r++)
    {
        matMulNaive<<<grid, block>>>(dA, dB, dC, NM);
    }

    CHECK(cudaEventRecord(stopEvent));
    CHECK(cudaEventSynchronize(stopEvent));

    float ms = 0.0f;

    CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));

    float tM = ms / reps;

    CHECK(cudaMemcpy(hC, dC, bytesM,
                     cudaMemcpyDeviceToHost));

    bool okM = checkMatMul(hA, hB, hC, NM);

    double gflops =
        2.0 * NM * NM * (double)NM / (tM * 1e6);

    printf("\n============================================\n");
    printf(" PART B: NAIVE MATRIX MULTIPLICATION (N = %d)\n", NM);
    printf("============================================\n");

    printf("Naive matmul: %8.3f ms | %7.1f GFLOPS | %s\n",
           tM, gflops, okM ? "PASS" : "FAIL");


    // ========================================================
    // SUMMARY
    // ========================================================

    printf("\n============================================\n");
    printf(" EXPERIMENT 6 SUMMARY\n");
    printf("============================================\n");

    printf("Transpose uncoalesced : %.3f ms\n", tN);
    printf("Transpose coalesced   : %.3f ms\n", tC);
    printf("Naive matmul          : %.3f ms\n", tM);

    printf("\nResult verification:\n");
    printf("Uncoalesced transpose : %s\n", okN ? "PASS" : "FAIL");
    printf("Coalesced transpose   : %s\n", okC ? "PASS" : "FAIL");
    printf("Naive matrix multiply : %s\n", okM ? "PASS" : "FAIL");


    // ========================================================
    // CLEANUP
    // ========================================================

    CHECK(cudaEventDestroy(startEvent));
    CHECK(cudaEventDestroy(stopEvent));

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));

    free(h_in);
    free(h_out);
    free(hA);
    free(hB);
    free(hC);

    printf("\nExperiment 6 completed.\n");

    return EXIT_SUCCESS;
}
