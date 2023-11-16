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

const int N = 1024;      // Número predeterm. de elementos en los vectores
const int CUDA_BLK = 16; // Tamaño predeterm. de bloque de hilos ƒCUDA

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
 * Función para inicializar los vectores que vamos a utilizar
 */
void Initialization(float A[], float B[], float C[], float D[], const unsigned int n)
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

// CPU execution
// ============================================================================

/**
 * Función PCR en la CPU
 */
void PCR_CPU(float X[], float Y[], float Z[], float W[], const unsigned int n)
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

    for (int j = 0; j < n / 2; j++)
    {
        float temp;
        temp = Y[j + n / 2] * Y[j] - Z[j] * X[j + n / 2];
        X[j] = (Y[j + n / 2] * W[j] - Z[j] * W[j + n / 2]) / temp;
        X[j + n / 2] = (W[j + n / 2] * Y[j] - W[j] * X[j + n / 2]) / temp;
    }

    for (int j = 0; j < n; j++)
    {
        printf(" \t %f  \n", X[j]);
    }
}

// GPU definition
// ============================================================================

extern __shared__ float array[];
__global__ void PCR_GPU_kernel(float *X, float *Y, float *Z,
                               float *W, const unsigned int n)
{
    unsigned int i, k;
    unsigned ln = floor(log2(float(n)));
    float alpha, gamma;

    int row = blockDim.y * blockIdx.y + threadIdx.y;

    float *Xs = (float *)array;
    float *Ys = (float *)&Xs[n];
    float *Zs = (float *)&Ys[n];
    float *Ws = (float *)&Zs[n];

    float Xr, Yr, Zr, Wr;

    if (row < n)
    {

        k = 1;
        for (i = 0; i < ln; i++)
        {
            Xs[threadIdx.y] = X[row];
            Ys[threadIdx.y] = Y[row];
            Zs[threadIdx.y] = Z[row];
            Ws[threadIdx.y] = W[row];
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
            X[row] = Xr;
            Y[row] = Yr;
            Z[row] = Zr;
            W[row] = Wr;
            //}
        }
    }
}

void PCR_GPU(float X[], float Y[], float Z[], float W[], const unsigned int n,
             const unsigned int block_size)
{
    float *d_X, *d_Y, *d_Z, *d_W;

    // Número de bytes a reservar para nuestros vectores
    unsigned int numBytes = n * sizeof(float);

    cudaMalloc(&d_X, numBytes);
    cudaMalloc(&d_Y, numBytes);
    cudaMalloc(&d_Z, numBytes);
    cudaMalloc(&d_W, numBytes);

    cudaMemcpy(d_X, X, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z, Z, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, numBytes, cudaMemcpyHostToDevice);

    // Launch kernel
    //  - threads_per_block: number of CUDA threads per grid block
    //	- blocks_in_grid   : number of blocks in grid
    //	(These are c structs with 3 member variables x, y, x)
    dim3 threads_per_block(1,
                           block_size,
                           1); // dim3 variable holds 3 dimensions
    dim3 blocks_in_grid(1,
                        1,
                        // ceil(float(n) / threads_per_block.y),
                        1);
    unsigned int sharedSize = 4 * numBytes;
    PCR_GPU_kernel<<<blocks_in_grid, threads_per_block, sharedSize>>>(d_X, d_Y, d_Z, d_W, n);

    // Copy data from device to CPU
    cudaMemcpy(X, d_X, numBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(Y, d_Y, numBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(Z, d_Z, numBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(W, d_W, numBytes, cudaMemcpyDeviceToHost);

    for (int j = 0; j < n / 2; j++)
    {
        float temp;
        temp = Y[j + n / 2] * Y[j] - Z[j] * X[j + n / 2];
        X[j] = (Y[j + n / 2] * W[j] - Z[j] * W[j + n / 2]) / temp;
        X[j + n / 2] = (W[j + n / 2] * Y[j] - W[j] * X[j + n / 2]) / temp;
    }

    for (int j = 0; j < n; j++)
    {
        printf(" \t %f  \n", X[j]);
    }

    // Free CPU and GPU memory
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_Z);
    cudaFree(d_W);
}

// Main program
// ============================================================================
/**
 * Función principal
 */
int main(int argc, char *argv[])
{
    // Para medir tiempos
    resnfo start, end, startgpu, endgpu;
    timenfo time, timegpu;

    // Aceptamos algunos parámetros

    // Número de elementos en los vectores (predeterminado: N)
    unsigned int n = (argc > 1) ? atoi(argv[1]) : N;
    unsigned int block_size = (argc > 2) ? atoi(argv[2]) : CUDA_BLK;

    // Número de bytes a reservar para nuestros vectores
    unsigned int numBytes = n * sizeof(float);

    // Reservamos e inicializamos vectores
    timestamp(&start);
    float *Av = (float *)malloc(numBytes);
    float *Bv = (float *)malloc(numBytes);
    float *Cv = (float *)malloc(numBytes);
    float *Dv = (float *)malloc(numBytes);
    Initialization(Av, Bv, Cv, Dv, n);
    timestamp(&end);
    myElapsedtime(start, end, &time);
    printtime(time);
    printf(" -> Reservar e inicializar vectores (%u)\n\n", n);

    // CPU execution
    timestamp(&start);
    PCR_CPU(Av, Bv, Cv, Dv, n);
    timestamp(&end);
    myElapsedtime(start, end, &time);
    printtime(time);
    printf(" -> PCR en la CPU\n\n");

    // GPU execution
    Initialization(Av, Bv, Cv, Dv, n);

    timestamp(&start);
    PCR_GPU(Av, Bv, Cv, Dv, n, block_size);
    timestamp(&end);
    myElapsedtime(start, end, &time);
    printtime(time);
    printf(" -> PCR en la GPU\n\n");

    free(Av);
    free(Bv);
    free(Cv);
    free(Dv);

    return (0);
}
