/**
 * Lecture 3 - Euclidean Vector Norm
 *
 * Implement a CUDA code that calculates the sumatory of the square for several
 * vectors on the GPU, which can be later used to calculate the euclidean norm
 * (if applying the square root)
 *
 * As input only one matrix of size "mxn" is created (floats). It represents
 * “m” vectors (rows) of length “n”
 *
 * As output you should provide a vector with “n” floats (element “i” is the
 * summatory of vector “i”)
 *
 * Each thread calculates one output element (the whole calculation for one vector)
 */
#include <stdio.h>

// Values for MxN matrix
const int DEFAULT_M = 100;
const int DEFAULT_N = 200;

const int DEFAULT_BLOCK_SIZE = 128; // Default CUDA block size

/**
 * Prints the values of the matrix to the screen
 */
void print_matrix(float *matrix, const unsigned int m, const unsigned int n)
{
    unsigned int i, j;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("%f ", matrix[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * Populate the matrix with values for the tests
*/
void populate_matrix(float *matrix, const unsigned int m, const unsigned int n)
{
    unsigned int i, j;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            matrix[i * n + j] = (i + 1) + j;
        }
    }
}

// CPU execution
// ============================================================================
void euclidean_vec_norm_CPU(float *a, float *c, const unsigned int m, const unsigned int n)
{
    unsigned int i, j;
    float res;
    for (i = 0; i < m; i++)
    {
        res = 0;
        for (j = 0; j < n; j++)
        {
            res += (a[i * n + j] * a[i * n + j]);
        }
        c[i] = res;
    }
}

// Kernel definition
// ============================================================================
__global__ void euclidean_vec_norm_GPU(float *a, float *c, const unsigned int m, const unsigned int n)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < m)
    {
        for (int i = 0; i < n; i++)
        {
            c[row] += (a[row * n + i] * a[row * n + i]);
        }
    }
}

// Main program
// ============================================================================
int main(int argc, char *argv[])
{
    // Read from args the matrix size
    unsigned int m = (argc > 1) ? atoi(argv[1]) : DEFAULT_M;
    unsigned int n = (argc > 2) ? atoi(argv[2]) : DEFAULT_N;
    unsigned int block_size = (argc > 3) ? atoi(argv[3]) : DEFAULT_BLOCK_SIZE;

    // Number of bytes to allocate for N vector
    size_t vectorNumBytes = m * sizeof(float);
    // Number of bytes to allocate for MxN matrix
    size_t matrixNumBytes = m * n * sizeof(float);

    float *h_A, *h_C, *l_A, *l_C, *d_A, *d_C;

    h_A = new float[m * n];
    h_C = new float[m];
    l_A = new float[m * n];
    l_C = new float[m];

    // Allocate device memory and copy input data over to GPU
    cudaMalloc(&d_A, matrixNumBytes);
    cudaMalloc(&d_C, vectorNumBytes);

    // Initialize host and local matrix A
    populate_matrix(h_A, m, n);
    populate_matrix(l_A, m, n);
    //print_matrix(h_A, m, n);
    //print_matrix(l_A, m, n);

    //  Initialize host and local array C for results
    for (int i = 0; i < m; i++)
    {
        h_C[i] = 0;
        l_C[i] = 0;
    }

    // Copy data from host matrix A to device matrix
    cudaMemcpy(d_A, h_A, matrixNumBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, vectorNumBytes, cudaMemcpyHostToDevice);

    // Launch kernel
    //  - threads_per_block: number of CUDA threads per grid block
    //	- blocks_in_grid   : number of blocks in grid
    //	(These are c structs with 3 member variables x, y, x)
    dim3 threads_per_block(1,
                           block_size,
                           1); // dim3 variable holds 3 dimensions
    dim3 blocks_in_grid(1,
                        ceil(float(m) / threads_per_block.y),
                        1);
    euclidean_vec_norm_GPU<<<blocks_in_grid, threads_per_block>>>(d_A, d_C, m, n);

    // Copy data from device to CPU
    cudaMemcpy(h_C, d_C, vectorNumBytes, cudaMemcpyDeviceToHost);

    euclidean_vec_norm_CPU(l_A, l_C, m, n);

    // Verify results
    for (int i = 0; i < m; i++)
        if (h_C[i] != l_C[i])
        {
            printf("Mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], l_C[i]);
            return -1;
        }

    // Free CPU and GPU memory
    cudaFree(d_A);
    cudaFree(d_C);

    printf("\n--------------------------------\n");
    printf("__SUCCESS__\n");
    printf("--------------------------------\n");
    printf("M                         = %d\n", m);
    printf("N                         = %d\n", n);
    printf("Threads Per Block (y-dim) = %d\n", threads_per_block.y);
    printf("Blocks In Grid (y-dim)    = %d\n", blocks_in_grid.y);
    printf("--------------------------------\n\n");

    return 0;
}
