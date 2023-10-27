#include <stdio.h>
#include <stdlib.h>

const int DEFAULT_NUM_BLK = 2;
const int DEFAULT_NUM_BLK_THREADS = 2;

__global__ void hello()
{
    printf("Block dimesion %u. Hello from block: %u, thread: %u\n", blockDim.x, blockIdx.x, threadIdx.x);
}

int main(int argc, char *argv[])
{
    // Cantidad de bloques
    unsigned int nBlocks = (argc > 1) ? atoi(argv[1]) : DEFAULT_NUM_BLK;

    // Cantidad de threads por bloque
    unsigned int nThreadsBlock = (argc > 2) ? atoi(argv[2]) : DEFAULT_NUM_BLK_THREADS;

    // Cantidad de bloques
    dim3 dimGrid(nBlocks);

    // Bloque unidimensional de hilos (*blk_size* hilos)
    dim3 dimBlock(nThreadsBlock);

    hello<<<dimGrid, dimBlock>>>();
    cudaDeviceSynchronize();
}