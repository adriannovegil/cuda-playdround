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

const int DSIZE = 4096;
const int block_size = 256; // CUDA maximum is 1024

// Kernel definition
// ============================================================================
__global__ void vadd(const float *A, const float *B, float *C, int ds)
{
    // create typical 1D thread index from built-in vari>
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx < ds)
    {
        C[idx] += A[idx] + B[idx]; // do the vector (element) add here
    }
}

// Main program
// ============================================================================
int main()
{
    // Number of bytes to allocate for an array
    unsigned int numBytes = DSIZE * sizeof(float);
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    // Allocate memory for arrays A, B, and C on host
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];

    // Allocate memory for arrays d_A, d_B, and d_C on device
    cudaErrorCheck(cudaMalloc(&d_A, numBytes));
    cudaErrorCheck(cudaMalloc(&d_B, numBytes));
    cudaErrorCheck(cudaMalloc(&d_C, numBytes));

    // Initialize host arrays A, B and C
    for (int i = 0; i < DSIZE; i++)
    { // initialize vectors in host memory
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = 0;
    }

    // Copy data from host arrays A and B to device arrays
    cudaErrorCheck(cudaMemcpy(d_A, h_A, numBytes, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(d_B, h_B, numBytes, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemset(d_C, 0, numBytes));

    // Launch kernel
    vadd<<<(DSIZE + block_size - 1) / block_size, block_size>>>(d_A, d_B, d_C, DSIZE);

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

    // Free CPU and GPU memory
    free(h_A);
    free(h_B);
    free(h_C);

    cudaErrorCheck(cudaFree(d_A));
    cudaErrorCheck(cudaFree(d_B));
    cudaErrorCheck(cudaFree(d_C));

    // Print the result
    printf("\n--------------------------------\n");
    printf("__SUCCESS__\n");
    printf("--------------------------------\n");
    printf("A[0] = %f\n", h_A[0]);
    printf("B[0] = %f\n", h_B[0]);
    printf("C[0] = %f\n", h_C[0]);
    printf("--------------------------------\n\n");

    return 0;
}
