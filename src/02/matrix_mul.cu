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

const int DSIZE = 8192;
const int block_size = 32; // CUDA maximum is 1024 *total* threads in block
const float A_val = 1.0f;
const float B_val = 2.0f;

// Kernel definition
// ============================================================================
__global__ void mmul(const float *A, const float *B, float *C, int ds)
{

    int column = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if ((column < ds) && (row < ds))
    {
        float temp = 0;
        for (int i = 0; i < ds; i++)
            temp += A[row * ds + i] * B[i * ds + column];
        C[row * ds + column] = temp;
    }
}

// Main program
// ============================================================================
int main()
{
    // Number of bytes to allocate for DSIZExDSIZE matrix
    size_t numBytes = DSIZE * DSIZE * sizeof(float);
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    h_A = new float[DSIZE * DSIZE];
    h_B = new float[DSIZE * DSIZE];
    h_C = new float[DSIZE * DSIZE];

    // Allocate device memory and copy input data over to GPU
    cudaErrorCheck(cudaMalloc(&d_A, numBytes));
    cudaErrorCheck(cudaMalloc(&d_B, numBytes));
    cudaErrorCheck(cudaMalloc(&d_C, numBytes));

    // Initialize host arrays A, B and C
    for (int i = 0; i < DSIZE * DSIZE; i++)
    {
        h_A[i] = A_val;
        h_B[i] = B_val;
        h_C[i] = 0;
    }

    // Copy data from host arrays A and B to device arrays
    cudaErrorCheck(cudaMemcpy(d_A, h_A, numBytes, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(d_B, h_B, numBytes, cudaMemcpyHostToDevice));

    // Launch kernel
    //  - threads_per_block: number of CUDA threads per grid block
    //	- blocks_in_grid   : number of blocks in grid
    //	(These are c structs with 3 member variables x, y, x)
    dim3 threads_per_block(block_size,
                           block_size, 1); // dim3 variable holds 3 dimensions
    dim3 blocks_in_grid((DSIZE + threads_per_block.x - 1) / threads_per_block.x,
                        (DSIZE + threads_per_block.y - 1) / threads_per_block.y, 1);
    mmul<<<blocks_in_grid, threads_per_block>>>(d_A, d_B, d_C, DSIZE);

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
    cudaErrorCheck(cudaMemcpy(h_C, d_C, numBytes, cudaMemcpyDeviceToHost));

    // Verify results
    for (int i = 0; i < DSIZE * DSIZE; i++)
        if (h_C[i] != A_val * B_val * DSIZE)
        {
            printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val * B_val * DSIZE);
            return -1;
        }

    // Free CPU and GPU memory
    cudaErrorCheck(cudaFree(d_A));
    cudaErrorCheck(cudaFree(d_B));
    cudaErrorCheck(cudaFree(d_C));

    printf("\n--------------------------------\n");
    printf("__SUCCESS__\n");
    printf("--------------------------------\n");
    printf("M                         = %d\n", DSIZE);
    printf("N                         = %d\n", DSIZE);
    printf("Threads Per Block (x-dim) = %d\n", threads_per_block.x);
    printf("Threads Per Block (y-dim) = %d\n", threads_per_block.y);
    printf("Blocks In Grid (x-dim)    = %d\n", blocks_in_grid.x);
    printf("Blocks In Grid (y-dim)    = %d\n", blocks_in_grid.y);
    printf("--------------------------------\n\n");

    return 0;
}
