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
#include <sys/time.h>
#include <sys/resource.h>

// Values for MxN matrix
const int DEFAULT_M = 100;
const int DEFAULT_N = 200;

const int DEFAULT_BLOCK_SIZE = 128; // Default CUDA block size
const float ZERO_VAL = 0.0f;

/**
 * Para medir el tiempo transcurrido (elapsed time):
 *
 * resnfo: tipo de dato definido para abstraer la métrica de recursos a usar
 * timenfo: tipo de dato definido para abstraer la métrica de tiempo a usar
 *
 * timestamp: abstrae función usada para tomar las muestras del tiempo transcurrido
 *
 * printtime: abstrae función usada para imprimir el tiempo transcurrido
 *
 * void myElapsedtime(resnfo start, resnfo end, timenfo *t): función para obtener
 * el tiempo transcurrido entre dos medidas
 */
#ifdef _noWALL_
typedef struct rusage resnfo;
typedef struct _timenfo
{
    double time;
    double systime;
} timenfo;
#define timestamp(sample) getrusage(RUSAGE_SELF, (sample))
#define printtime(t) printf("%15f s (%f user + %f sys) ", \
                            t.time + t.systime, t.time, t.systime);
#else
typedef struct timeval resnfo;
typedef double timenfo;
#define timestamp(sample) gettimeofday((sample), 0)
#define printtime(t) printf("%15f s ", t);
#endif

void myElapsedtime(const resnfo start, const resnfo end, timenfo *const t)
{
#ifdef _noWALL_
    t->time = (end.ru_utime.tv_sec + (end.ru_utime.tv_usec * 1E-6)) - (start.ru_utime.tv_sec + (start.ru_utime.tv_usec * 1E-6));
    t->systime = (end.ru_stime.tv_sec + (end.ru_stime.tv_usec * 1E-6)) - (start.ru_stime.tv_sec + (start.ru_stime.tv_usec * 1E-6));
#else
    *t = (end.tv_sec + (end.tv_usec * 1E-6)) - (start.tv_sec + (start.tv_usec * 1E-6));
#endif /*_noWALL_*/
}

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

/**
 * Prints the values of the array to the screen
*/
void print_array(float *array, const unsigned int m)
{
    unsigned int i;
    for (i = 0; i < m; i++)
    {
        printf("%f ", array[i]);
    }
    printf("\n");
}

/**
 * Populate the array with the value
 */
void populate_array(float *array, const unsigned int m, const float value)
{
    unsigned int i;
    for (i = 0; i < m; i++)
    {
        array[i] = value;
    }
}

/**
 * Compare the result vectors
 */
bool check_result(float *cpuArray, float *gpuArray, const unsigned int m)
{
    for (int i = 0; i < m; i++)
        if (cpuArray[i] != gpuArray[i])
        {
            printf("Mismatch at index %d, was: %f, should be: %f\n", i, gpuArray[i], cpuArray[i]);
            return false;
        }
    return true;
}

// CPU execution
// ============================================================================

void euclidean_vec_norm_CPU(float *a, float *c, const unsigned int m, const unsigned int n)
{
    unsigned int i, j;
    float res;
    resnfo start, end;
    timenfo time;

    timestamp(&start); // Start time measurement
    for (i = 0; i < m; i++)
    {
        res = 0;
        for (j = 0; j < n; j++)
        {
            res += (a[i * n + j] * a[i * n + j]);
        }
        c[i] = res;
    }
    timestamp(&end); // Stop time measurement
    myElapsedtime(start, end, &time);
    printtime(time);
}

// GPU definition
// ============================================================================

/**
 * Kernel definition
 */
__global__ void euclidean_vec_norm_GPU_kernel(float *a, float *c, const unsigned int m, const unsigned int n)
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

void euclidean_vec_norm_GPU(float *a, float *c, const unsigned int m,
                            const unsigned int n, const unsigned int block_size)
{
    resnfo start, end;
    timenfo time;

    float *d_A, *d_C;

    // Number of bytes to allocate for N vector
    size_t vectorNumBytes = m * sizeof(float);
    // Number of bytes to allocate for MxN matrix
    size_t matrixNumBytes = m * n * sizeof(float);

    // Allocate device memory and copy input data over to GPU
    cudaMalloc(&d_A, matrixNumBytes);
    cudaMalloc(&d_C, vectorNumBytes);

    // Copy data from host matrix A to device matrix
    cudaMemcpy(d_A, a, matrixNumBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, c, vectorNumBytes, cudaMemcpyHostToDevice);

    // Launch kernel
    //  - threads_per_block: number of CUDA threads per grid block
    //	- blocks_in_grid   : number of blocks in grid
    //	(These are c structs with 3 member variables x, y, x)
    dim3 threads_per_block(1,
                           block_size,
                           1); // dim3 variable holds 3 dimensions
    dim3 blocks_in_grid(1,
                        // ceil(float(m) / threads_per_block.y),
                        (m * m + threads_per_block.y - 1) / threads_per_block.y,
                        1);

    timestamp(&start); // Start time measurement
    euclidean_vec_norm_GPU_kernel<<<blocks_in_grid, threads_per_block>>>(d_A, d_C, m, n);
    cudaDeviceSynchronize();
    timestamp(&end); // Stop time measurement
    myElapsedtime(start, end, &time);
    printtime(time);

    // Copy data from device to CPU
    cudaMemcpy(c, d_C, vectorNumBytes, cudaMemcpyDeviceToHost);

    // Free CPU and GPU memory
    cudaFree(d_A);
    cudaFree(d_C);
}

// Main program
// ============================================================================
int main(int argc, char *argv[])
{
    float *h_A, *h_C, *l_A, *l_C;

    // Read from args the matrix size
    unsigned int m = (argc > 1) ? atoi(argv[1]) : DEFAULT_M;
    unsigned int n = (argc > 2) ? atoi(argv[2]) : DEFAULT_N;
    unsigned int block_size = (argc > 3) ? atoi(argv[3]) : DEFAULT_BLOCK_SIZE;

    h_A = new float[m * n];
    h_C = new float[m];
    l_A = new float[m * n];
    l_C = new float[m];

    // Initialize host and local matrix A
    populate_matrix(h_A, m, n);
    populate_matrix(l_A, m, n);

    // Initialize host and local array C for results
    populate_array(h_C, m, ZERO_VAL);
    populate_array(l_C, m, ZERO_VAL);

    // GPU Execution
    euclidean_vec_norm_GPU(h_A, h_C, m, n, block_size);
    printf(" -> Calculate the Euclidean vector norm in the GPU (%d vectos, %d elements)\n\n", m, n);

    // CPU execution
    euclidean_vec_norm_CPU(l_A, l_C, m, n);
    printf(" -> Calculate the Euclidean vector norm in the CPU (%d vectos, %d elements)\n\n", m, n);

    // Verify results
    if (!check_result(l_C, h_C, m))
    {
        return -1;
    }

    printf("\n--------------------------------\n");
    printf("__SUCCESS__\n");
    printf("--------------------------------\n");
    printf("M                         = %d\n", m);
    printf("N                         = %d\n", n);
    printf("--------------------------------\n\n");

    return 0;
}
