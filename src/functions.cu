#include <stdio.h>
#include <mpi.h>

#include "functions.h"
#include "utils.h"

cudaDeviceProp set_gpu_and_get_properties(int rank) {
    cudaDeviceProp prop;
    int device_count, device;

    CUDA_CHECK(cudaGetDeviceCount(&device_count), rank);

    if (device_count == 0) {
        fprintf(stderr, "Rank %d Error in get_gpu_properties: No CUDA-capable devices found.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    device = 0;
    CUDA_CHECK(cudaSetDevice(device), rank);
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device), rank);
    
    return prop;
}

int calculate_optimal_tile_width(cudaDeviceProp prop, int rank) {
    int tile_width;

    int max_threads_per_block_sqrt = (int)sqrt((double)prop.maxThreadsPerBlock);

    for (tile_width = MIN(prop.warpSize, max_threads_per_block_sqrt); tile_width >= 4; tile_width--) {
        int threads_per_block = tile_width * tile_width;
        
        size_t required_shared_memory = 2 * threads_per_block * sizeof(double);

        if (required_shared_memory <= prop.sharedMemPerBlock) {
            return tile_width;
        }
    }

    fprintf(stderr, "Rank %d: Error: Unable to determine a suitable tile_width.\n", rank);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
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

    int phase;
    double c_value = 0;
    for(phase = 0; phase < ceil(K/(float)tile_width); ++phase) {
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