#include <stdio.h>

#include "functions.h"

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

cudaDeviceProp get_gpu_properties() {
    cudaDeviceProp prop;
    int device_count;
    cudaError_t err;

    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in get_gpu_properties: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE); 
    }
    
    if (device_count == 0) {
        fprintf(stderr, "Error in get_gpu_properties: No CUDA-capable devices found.\n");
        exit(EXIT_FAILURE);
    }

    err = cudaGetDeviceProperties(&prop, 0); 
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in get_gpu_properties: cudaGetDeviceProperties for device 0 failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    return prop;
}

int calculate_optimal_tile_width(cudaDeviceProp prop) {
    int max_threads_per_block_sqrt = (int)sqrt((double)prop.maxThreadsPerBlock);

    for (int tile_width = MIN(prop.warpSize, max_threads_per_block_sqrt); tile_width >= 4; tile_width--) {
        int threads_per_block = tile_width * tile_width;
        
        size_t required_shared_memory = 2 * threads_per_block * sizeof(double);

        if (required_shared_memory <= prop.sharedMemPerBlock) {
            printf("Selected tile_width = %d\n", tile_width);
            printf("Threads per block = %d (Max: %d)\n", threads_per_block, prop.maxThreadsPerBlock);
            printf("Shared memory per block required = %zu bytes (Max: %zu)\n", required_shared_memory, prop.sharedMemPerBlock);
            return tile_width;
        }
    }

    fprintf(stderr, "Error: Unable to determine a suitable tile_width.\n");
    exit(EXIT_FAILURE);
    return 0;
}


/**
 * Performs matrix multiplication C = A * B using a tiled approach
 * with shared memory to optimize global memory accesses.
 * 
 * Each thread block computes one tile of matrix C.
 * Within each block, threads cooperatively load the corresponding tiles
 * from matrices A and B into shared memory.
 * The multiplication is then performed using data from shared memory.
 * The process is iterated through "phases" to cover the entire K dimension.
 *
 * A Pointer to matrix A (M x K) in global memory.
 * B Pointer to matrix B (K x N) in global memory.
 * C Pointer to the resulting matrix C (M x N) in global memory.
 * M Number of rows in matrix A and matrix C.
 * N Number of columns in matrix B and matrix C.
 * K Number of columns in matrix A and number of rows in matrix B.
 */
__global__ void matrix_mul_kernel(double* A, double* B, double* C, int M, int N, int K) {

    extern __shared__ double shared_mem[];

    int tile_width = blockDim.x;

    double* s_A = (double*) shared_mem;
    double* s_B = (double*) shared_mem + tile_width * tile_width;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * tile_width + ty;
    int col = bx * tile_width + tx;

    double c_value = 0;
    for(int phase = 0; phase < ceil(K/(float)tile_width); ++phase) {
        if((row < M) && (phase * tile_width + tx) < K)
            s_A[ty * tile_width + tx] = A[row * K + phase * tile_width + tx];
        else
            s_A[ty * tile_width + tx] = 0.0;

        if((phase * tile_width + ty) < K && (col < N))
            s_B[ty * tile_width + tx] = B[(phase * tile_width + ty) * N + col];
        else
            s_B[ty * tile_width + tx] = 0.0;
        
        __syncthreads();

        for (int k = 0; k < tile_width; ++k) {
            c_value += s_A[ty * tile_width + k] * s_B[k * tile_width + tx];
        }
        __syncthreads();
    }

    if ((row < M) && (col < N))
        C[row * N + col] = c_value;
}