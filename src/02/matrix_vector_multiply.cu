#include <stdio.h>

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                                     \
    do                                                                                           \
    {                                                                                            \
        cudaError_t cuErr = call;                                                                \
        if (cudaSuccess != cuErr)                                                                \
        {                                                                                        \
            printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr)); \
            exit(0);                                                                             \
        }                                                                                        \
    } while (0)

// Values for MxN matrix
#define M 100
#define N 200

// Kernel definition
// ============================================================================
__global__ void multiply_mat_vec(int *a, int *x, int *y)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < M)
    {
        int thread_id = row;
        for (int i = 0; i < N; i++)
        {
            y[thread_id] = y[thread_id] + a[row * N + i] * x[i];
        }
    }
}

// Main program
// ============================================================================
int main()
{
    // Number of bytes to allocate for MxN matrix
    size_t bytes = M * N * sizeof(int);
    int *d_A;
    int *d_x, *d_y;

    // Allocate memory for arrays A, x, and y on host
    int A[M][N];
    int x[N], y[M];

    // Allocate memory for arrays d_A, d_x, and d_y on device
    cudaErrorCheck(cudaMalloc(&d_A, bytes));
    cudaErrorCheck(cudaMalloc(&d_x, N * sizeof(int)));
    cudaErrorCheck(cudaMalloc(&d_y, M * sizeof(int)));

    // Initialize host matrix A
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i][j] = 1;
        }
    }

    // Initialize host array x
    for (int i = 0; i < N; i++)
    {
        x[i] = 1;
    }

    // Initialize host array y
    for (int i = 0; i < M; i++)
    {
        y[i] = 0;
    }

    // Copy data from host arrays A and x to device arrays
    cudaErrorCheck(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(d_x, x, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    // Set execution configuration parameters
    //    threads_per_block: number of CUDA threads per grid block
    //    blocks_in_grid   : number of blocks in grid
    //    (These are c structs with 3 member variables x, y, x)
    //
    // 16*16 = 256 thread per block
    dim3 threads_per_block(1, 128, 1);
    dim3 blocks_in_grid(1,
                        ceil(float(M) / threads_per_block.y),
                        1);
    multiply_mat_vec<<<blocks_in_grid, threads_per_block>>>(d_A, d_x, d_y);

    // Check for errors in kernel launch (e.g. invalid execution configuration paramters)
    cudaError_t cuErrSync = cudaGetLastError();
    if (cuErrSync != cudaSuccess)
    {
        printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErrSync));
        exit(0);
    }

    // Check for errors on the GPU after control is returned to CPU
    cudaError_t cuErrAsync = cudaDeviceSynchronize();
    if (cuErrAsync != cudaSuccess)
    {
        printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErrAsync));
        exit(0);
    }

    // Copy data from device to CPU
    cudaErrorCheck(cudaMemcpy(y, d_y, M * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify results
    for (int i = 0; i < M; i++)
    {
        if (y[i] != N)
        {
            printf("Error - y[%d] = %d instead of %d\n", i, y[i], N);
            exit(0);
        }
    }

    // Free CPU and GPU memory
    cudaErrorCheck(cudaFree(d_A));
    cudaErrorCheck(cudaFree(d_x));
    cudaErrorCheck(cudaFree(d_y));

    printf("\n--------------------------------\n");
    printf("__SUCCESS__\n");
    printf("--------------------------------\n");
    printf("M                         = %d\n", M);
    printf("N                         = %d\n", N);
    printf("Threads Per Block (x-dim) = %d\n", threads_per_block.x);
    printf("Threads Per Block (y-dim) = %d\n", threads_per_block.y);
    printf("Blocks In Grid (x-dim)    = %d\n", blocks_in_grid.x);
    printf("Blocks In Grid (y-dim)    = %d\n", blocks_in_grid.y);
    printf("--------------------------------\n\n");

    return 0;
}