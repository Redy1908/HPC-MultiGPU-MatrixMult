#include <common_functions.h>
#include <math.h>
#include <stdlib.h>

#include "functions.cuh"
#include "utils.h"

cudaDeviceProp set_gpu_and_get_properties(int rank) {
  cudaDeviceProp prop;
  int device_count, device;

  CUDA_CHECK(cudaGetDeviceCount(&device_count), rank);

  if (device_count == 0) {
    fprintf(
        stderr,
        "Rank %d Error in get_gpu_properties: No CUDA-capable devices found.\n",
        rank);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  device = 0;
  CUDA_CHECK(cudaSetDevice(device), rank);
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device), rank);

  return prop;
}

void check_threads_per_block(cudaDeviceProp prop, int tile_width, int rank) {
  if (rank == 0) {
    int threads_per_block = tile_width * tile_width;

    if (threads_per_block > prop.maxThreadsPerBlock) {
      fprintf(stderr,
              "Rank %d: Error: Threads per block (%d) exceeds GPU max threads per block (%d).\n",
              rank, threads_per_block, prop.maxThreadsPerBlock);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }
}

void check_shared_memory_usage(cudaDeviceProp prop, int tile_width, int rank) {
  if (rank == 0) {
    int threads_per_block = tile_width * tile_width;

    int required_shared_memory_size = 2 * threads_per_block * sizeof(double);

    if (required_shared_memory_size > prop.sharedMemPerBlock) {
      fprintf(stderr,
              "Rank %d: Warning: Required shared memory size (%d bytes) exceeds "
              "available shared memory (%zu bytes) per block. Performance will be affected.\n",
              rank, required_shared_memory_size, prop.sharedMemPerBlock);
    }
  }
}

int SUMMA(MPI_Comm grid_comm, double *A, double *B, double *C, uint m, uint k, uint n, dim3 grid_size, dim3 block_size) {
  int dims[2], periods[2], coords[2];
  MPI_Cart_get(grid_comm, 2, dims, periods, coords);

  int K2 = find_lcm(dims[0], dims[1]);

  uint a_block_height = m / dims[0];
  uint a_block_width = k / K2;

  uint b_block_height = a_block_width;
  uint b_block_width = n / dims[1];

  double *a_block = (double *)malloc(a_block_height * a_block_width * sizeof(double));
  double *b_block = (double *)malloc(b_block_height * b_block_width * sizeof(double));

  double *A_dev, *B_dev, *C_dev;
  cudaMalloc(&A_dev, a_block_height * a_block_width * sizeof(double));
  cudaMalloc(&B_dev, b_block_height * b_block_width * sizeof(double));
  cudaMalloc(&C_dev, a_block_height * b_block_width * sizeof(double));

  if (a_block == NULL || b_block == NULL || A_dev == NULL || B_dev == NULL || C_dev == NULL) {
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    free(a_block);
    free(b_block);

    return 1;
  }

  /* create communicators along rows and columns */
  int remain_dims_row[2] = {0, 1};
  int remain_dims_col[2] = {1, 0};
  MPI_Comm row_comm, col_comm;
  MPI_Cart_sub(grid_comm, remain_dims_row, &row_comm);
  MPI_Cart_sub(grid_comm, remain_dims_col, &col_comm);

  double *A_start = A;
  double *B_start = B;

  for (uint i = 0; i < K2; i++) {
    uint r = i % dims[0];
    uint c = i % dims[1];

    if (coords[1] == c) {
      for (uint j = 0; j < a_block_height; j++)
        memcpy(a_block + j * a_block_width, A_start + j * i, a_block_width * sizeof(double));

      A_start += a_block_width;
    }

    if (coords[0] == r) {
      for (uint j = 0; j < b_block_height; j++)
        memcpy(b_block + j * b_block_width, A_start + j * i, b_block_width * sizeof(double));

      B_start += b_block_width * n;
    }

    MPI_Bcast(a_block, a_block_height * a_block_width, MPI_DOUBLE, c, row_comm);
    MPI_Bcast(b_block, b_block_height * b_block_width, MPI_DOUBLE, r, col_comm);

    /* compute submatrix multiplication on the GPU */
    uint shared_mem_size = 2 * block_size.x * block_size.y * sizeof(double);
    cudaMemcpy(A_dev, a_block, a_block_height * a_block_width * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, b_block, b_block_height * b_block_width * sizeof(double), cudaMemcpyHostToDevice);
    matrix_mul_kernel<<<grid_size, block_size, shared_mem_size>>>(A_dev, B_dev, C_dev, a_block_height, b_block_width, a_block_width);
    cudaDeviceSynchronize(); /* TODO: is this needed? */
  }

  /* copy the final result from the GPU to the CPU */
  cudaDeviceSynchronize();
  cudaMemcpy(C, C_dev, a_block_height * b_block_width * sizeof(double), cudaMemcpyDeviceToHost);

  /* cleanup */
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);

  cudaFree(A_dev);
  cudaFree(B_dev);
  cudaFree(C_dev);

  free(a_block);
  free(b_block);

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
__global__ void matrix_mul_kernel(double *A, double *B, double *C, int M, int N, int K) {
  extern __shared__ double shared_mem[];

  int tile_width = blockDim.x;

  double *s_A = (double *)shared_mem;
  double *s_B = (double *)shared_mem + tile_width * tile_width;

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * tile_width + ty;
  int col = bx * tile_width + tx;

  double c_value = 0.0;
  for (int phase = 0; phase < ceil(K / (float)tile_width); ++phase) {
    if ((row < M) && (phase * tile_width + tx) < K)
      s_A[ty * tile_width + tx] = A[row * K + phase * tile_width + tx];
    else
      s_A[ty * tile_width + tx] = 0.0;

    if ((phase * tile_width + ty) < K && (col < N))
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
    C[row * N + col] += c_value;
}