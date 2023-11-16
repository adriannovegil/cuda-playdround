/**
 * Lecture 4
 *
 * Programación de GPUs (General Purpose Computation on Graphics Processing
 * Unit)
 *
 * PCR en GPU
 * Parámetros opcionales (en este orden): sumavectores #rep #n #blk
 * #rep: número de repetiones
 * #n: número de elementos en cada vector
 * #blk: hilos por bloque CUDA
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>

const int N = 1024; // Número predeterm. de elementos en los vectores
// const int CUDA_BLK = 16;             // Tamaño predeterm. de bloque de hilos ƒCUDA
const int NUMBER_OF_SYSTEMS = 32760; // Cantidad de sistemas a calcular

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
 * Prints the values of the array to the screen
 */
void print_array(float *array, const unsigned int m)
{
    unsigned int i;
    for (i = 0; i < m; i++)
    {
        printf("%f ", array[i]);
    }
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
 * Function that comprate the elements of two array's
 */
bool compare_array(float *a1, float *a2, const unsigned int m)
{
    for (int i = 0; i < m; i++)
        if (a1[i] != a2[i])
        {
            printf("Mismatch at index %d, was: %f, should be: %f\n", i, a1[i], a2[i]);
            return false;
        }
    return true;
}

/**
 * Función para inicializar los vectores que vamos a utilizar
 */
void systemInitialization(float A[], float B[], float C[], float D[], const unsigned int n)
{
    unsigned int i;

    A[0] = 0.0;
    B[0] = 2.0;
    C[0] = -1.0;
    D[0] = 1.0;

    for (i = 1; i < n - 1; i++)
    {
        A[i] = -1.0;
        B[i] = 2.0;
        C[i] = -1.0;
        D[i] = 0.0;
    }

    A[n - 1] = -1.0;
    B[n - 1] = 2.0;
    C[n - 1] = 0.0;
    D[n - 1] = 1.0;
}

/**
 * Función que inicializa la matriz de sistemas
 */
void systemsInitialization(float A[], float B[], float C[], float D[],
                           const unsigned int nSystems,
                           const unsigned int nElements)
{
    unsigned int system;
    for (system = 0; system < nSystems; system++)
    {
        systemInitialization(&A[(system * nElements)],
                             &B[(system * nElements)],
                             &C[(system * nElements)],
                             &D[(system * nElements)],
                             nElements);
    }
}

/**
 * Función que calcula el resutlado final del sistema
 */
void calculateResult(float A[], float B[], float C[], float D[], const unsigned int n)
{
    for (int j = 0; j < n / 2; j++)
    {
        float temp;
        temp = B[j + n / 2] * B[j] - C[j] * A[j + n / 2];
        A[j] = (B[j + n / 2] * D[j] - C[j] * D[j + n / 2]) / temp;
        A[j + n / 2] = (D[j + n / 2] * B[j] - D[j] * A[j + n / 2]) / temp;
    }
}

// CPU execution
// ============================================================================

/**
 *
 */
void pcr_cpu_kernel(float X[], float Y[], float Z[], float W[], const unsigned int n)
{
    unsigned int i, k;
    unsigned ln = floor(log2(float(n)));
    float alpha, gamma;

    unsigned int numBytes = n * sizeof(float);

    float *Xr = (float *)malloc(numBytes);
    float *Yr = (float *)malloc(numBytes);
    float *Zr = (float *)malloc(numBytes);
    float *Wr = (float *)malloc(numBytes);

    k = 1;
    for (i = 0; i < ln; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (j >= k)
            {
                if (j <= (n - k - 1))
                {
                    alpha = -X[j] / Y[j - k];
                    gamma = -Z[j] / Y[j + k];
                    Yr[j] = Y[j] + (alpha * Z[j - k] + gamma * X[j + k]);
                    Xr[j] = alpha * X[j - k];
                    Zr[j] = gamma * Z[j + k];
                    Wr[j] = W[j] + (alpha * W[j - k] + gamma * W[j + k]);
                }
                else
                {
                    alpha = -X[j] / Y[j - k];
                    Yr[j] = Y[j] + (alpha * Z[j - k]);
                    Xr[j] = alpha * X[j - k];
                    Zr[j] = 0;
                    Wr[j] = W[j] + (alpha * W[j - k]);
                }
            }
            else
            {
                gamma = -Z[j] / Y[j + k];
                Yr[j] = Y[j] + gamma * X[j + k];
                Xr[j] = 0;
                Zr[j] = gamma * Z[j + k];
                Wr[j] = W[j] + gamma * W[j + k];
            }
        }
        k = k << 1;
        for (int j = 0; j < n; j++)
        {
            X[j] = Xr[j];
            Y[j] = Yr[j];
            Z[j] = Zr[j];
            W[j] = Wr[j];
        }
    }
}

/**
 * Función PCR en la CPU
 */
void pcr_cpu(const unsigned int n)
{
    // Para medir tiempos
    resnfo start, end;
    timenfo time;

    // Número de bytes a reservar para nuestros vectores
    unsigned int numBytes = n * sizeof(float);

    // Reservamos e inicializamos vectores
    timestamp(&start);
    float *h_A = (float *)malloc(numBytes);
    float *h_B = (float *)malloc(numBytes);
    float *h_C = (float *)malloc(numBytes);
    float *h_D = (float *)malloc(numBytes);
    systemInitialization(h_A, h_B, h_C, h_D, n);
    timestamp(&end);
    myElapsedtime(start, end, &time);
    printtime(time);
    printf(" -> Reservar e inicializar vectores CPU: (%u) elementos\n\n", n);

    // CPU execution
    timestamp(&start);
    pcr_cpu_kernel(h_A, h_B, h_C, h_D, n);
    timestamp(&end);
    myElapsedtime(start, end, &time);
    printtime(time);
    printf(" -> PCR en la CPU\n\n");

    calculateResult(h_A, h_B, h_C, h_D, n);

    //for (int j = 0; j < n; j++)
    //{
    //    printf(" \t %f  \n", h_A[j]);
    //}

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
}

// GPU execution
// ============================================================================

/**
 * Kernel definition
 */
extern __shared__ float array[];
__global__ void pcr_gpu_kernel(float *X, float *Y, float *Z, float *W,
                               const unsigned int number_of_systems,
                               const unsigned int n)
{
    unsigned int i, k;
    unsigned ln = floor(log2(float(n)));
    float alpha, gamma;

    int global_pos = blockDim.y * blockIdx.y + threadIdx.y;
    int row = threadIdx.y;

    float *Xs = (float *)array;
    float *Ys = (float *)&Xs[n];
    float *Zs = (float *)&Ys[n];
    float *Ws = (float *)&Zs[n];

    float Xr, Yr, Zr, Wr;

    if (global_pos < number_of_systems * n)
    {
        k = 1;
        for (i = 0; i < ln; i++)
        {
            Xs[threadIdx.y] = X[global_pos];
            Ys[threadIdx.y] = Y[global_pos];
            Zs[threadIdx.y] = Z[global_pos];
            Ws[threadIdx.y] = W[global_pos];
            // We synchronize threads to ensure the loading of the entire sub-array
            __syncthreads();

            // for (int j = 0; j < n; j++)
            //{
            if (row >= k)
            {
                if (row <= (n - k - 1))
                {
                    alpha = -Xs[row] / Ys[row - k];
                    gamma = -Zs[row] / Ys[row + k];
                    Yr = Ys[row] + (alpha * Zs[row - k] + gamma * Xs[row + k]);
                    Xr = alpha * Xs[row - k];
                    Zr = gamma * Zs[row + k];
                    Wr = Ws[row] + (alpha * Ws[row - k] + gamma * Ws[row + k]);
                }
                else
                {
                    alpha = -Xs[row] / Ys[row - k];
                    Yr = Ys[row] + (alpha * Zs[row - k]);
                    Xr = alpha * Xs[row - k];
                    Zr = 0;
                    Wr = Ws[row] + (alpha * Ws[row - k]);
                }
            }
            else
            {
                gamma = -Zs[row] / Ys[row + k];
                Yr = Ys[row] + gamma * Xs[row + k];
                Xr = 0;
                Zr = gamma * Zs[row + k];
                Wr = Ws[row] + gamma * Ws[row + k];
            }
            //}

            __syncthreads();

            k = k << 1;

            // for (int j = 0; j < n; j++)
            //{
            X[global_pos] = Xr;
            Y[global_pos] = Yr;
            Z[global_pos] = Zr;
            W[global_pos] = Wr;
            //}
        }
    }
}

/**
 * Función PCR en la GPU
 */
void pcr_gpu(const unsigned int number_of_systems,
             const unsigned int n)
{
    // Para medir tiempos
    resnfo startgpu, endgpu;
    timenfo timegpu;

    // Número de bytes a reservar para nuestros vectores
    unsigned int numBytes = number_of_systems * n * sizeof(float);

    // Reservamos e inicializamos vectores
    timestamp(&startgpu);
    float *d_A, *d_B, *d_C, *d_D;
    float *A = (float *)malloc(numBytes);
    float *B = (float *)malloc(numBytes);
    float *C = (float *)malloc(numBytes);
    float *D = (float *)malloc(numBytes);
    systemsInitialization(A, B, C, D, number_of_systems, n);
    cudaMalloc(&d_A, numBytes);
    cudaMalloc(&d_B, numBytes);
    cudaMalloc(&d_C, numBytes);
    cudaMalloc(&d_D, numBytes);
    cudaMemcpy(d_A, A, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, D, numBytes, cudaMemcpyHostToDevice);
    timestamp(&endgpu);
    myElapsedtime(startgpu, endgpu, &timegpu);
    printtime(timegpu);
    printf(" -> Reservar e inicializar vectores GPU: (%u) sistemas de (%u) elementos\n\n", number_of_systems, n);

    // GPU Execution
    //  - threads_per_block: number of CUDA threads per grid block
    //	- blocks_in_grid   : number of blocks in grid
    //	(These are c structs with 3 member variables x, y, x)
    dim3 threads_per_block(1,
                           n,
                           // block_size,
                           1); // dim3 variable holds 3 dimensions
    dim3 blocks_in_grid(1,
                        number_of_systems,
                        // ceil(float(n) / threads_per_block.y),
                        1);
    unsigned int sharedSize = n * sizeof(float) * 4;
    timestamp(&startgpu);
    cudaFuncSetCacheConfig(pcr_gpu_kernel, cudaFuncCachePreferShared);
    pcr_gpu_kernel<<<blocks_in_grid, threads_per_block, sharedSize>>>(d_A, d_B, d_C, d_D, number_of_systems, n);
    cudaDeviceSynchronize();
    timestamp(&endgpu);
    myElapsedtime(startgpu, endgpu, &timegpu);
    printtime(timegpu);
    printf(" -> PCR en la GPU\n\n");

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
    cudaMemcpy(A, d_A, numBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(B, d_B, numBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(C, d_C, numBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(D, d_D, numBytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < number_of_systems; i++)
    {
        calculateResult(&A[(i * n)],
                        &B[(i * n)],
                        &C[(i * n)],
                        &D[(i * n)],
                        n);
    }

    //printf(" Av= [");
    //print_matrix(A, number_of_systems, n);
    //printf("]\n\n");

    // Free CPU and GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    free(A);
    free(B);
    free(C);
    free(D);
}

// Main program
// ============================================================================

/**
 * Función principal
 */
int main(int argc, char *argv[])
{
    // Read program arguments
    unsigned int n = (argc > 1) ? atoi(argv[1]) : N;
    unsigned int number_of_systems = (argc > 2) ? atoi(argv[2]) : NUMBER_OF_SYSTEMS;

    printf("\n--------------------------------\n");
    printf(" Parallel Cyclic Reduction (PCR)\n");
    printf("--------------------------------\n");
    printf(" Number of systems         = %d\n", number_of_systems);
    printf(" System size               = %d\n", n);
    printf(" Execution time            = \n\n");

    if((n <= 0) || !((n &(n - 1)) == 0))
    {
        printf(" Error - The N value have to be >0 and power of 2\n");
        return -1;
    }

    // Llamada a la función d ejecución de la CPU
    pcr_cpu(n);

    // Llamada a la función d ejecución de la GPU
    pcr_gpu(number_of_systems, n);

    printf("--------------------------------\n");
    printf(" SUCCESS\n");
    printf("--------------------------------\n\n");

    return (0);
}
