#include <math.h>
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

void SUMMA(MPI_Comm grid_comm, double *A, double *B, double *C_host_block, int M, int K, int N, int tile_width, int rank) {

  int dims[2], periods[2], coords[2];
  int i, j, k, c, r, K2;

  int block_rows_A, block_rows_B;
  int block_cols_A, block_cols_B;

  double *A_col_host, *B_row_host, *A_start, *B_start;
  double *d_A_col, *d_B_row, *d_C_block;

  MPI_Comm row_comm, col_comm;

  int remain_dims_row[2] = {0, 1};
  int remain_dims_col[2] = {1, 0};

  MPI_Cart_get(grid_comm, 2, dims, periods, coords);

  K2 = find_lcm(dims[0], dims[1]);

  block_rows_A = M / dims[0];
  block_rows_B = K / K2;
  block_cols_A = K / K2;
  block_cols_B = N / dims[1];

  A_col_host = (double *)malloc(block_rows_A * block_cols_A * sizeof(double));
  MALLOC_CHECK(A_col_host, rank, "A_col_host");

  B_row_host = (double *)malloc(block_rows_B * block_cols_B * sizeof(double));
  MALLOC_CHECK(B_row_host, rank, "B_row_host");

  CUDA_CHECK(cudaMalloc((void **)&d_A_col, block_rows_A * block_cols_A * sizeof(double)), rank);
  CUDA_CHECK(cudaMalloc((void **)&d_B_row, block_rows_B * block_cols_B * sizeof(double)), rank);
  CUDA_CHECK(cudaMalloc((void **)&d_C_block, block_rows_A * block_cols_B * sizeof(double)), rank);

  CUDA_CHECK(cudaMemset(d_C_block, 0, block_rows_A * block_cols_B * sizeof(double)), rank);

  MPI_Cart_sub(grid_comm, remain_dims_row, &row_comm);
  MPI_Cart_sub(grid_comm, remain_dims_col, &col_comm);

  A_start = A;
  B_start = B;

  for (k = 0; k < K2; k++) {
    c = k % dims[1];
    r = k % dims[0];

    if (coords[1] == c) {
      for (i = 0; i < block_rows_A; i++) {
        for (j = 0; j < block_cols_A; j++) {
          A_col_host[i * block_cols_A + j] = A_start[i * K + j];
        }
      }
      A_start += block_cols_A;
    }

    if (coords[0] == r) {
      for (i = 0; i < block_rows_B; i++) {
        for (j = 0; j < block_cols_B; j++) {
          B_row_host[i * block_cols_B + j] = B_start[i * N + j];
        }
      }
      B_start += block_cols_B * N;
    }

    MPI_Bcast(A_col_host, block_rows_A * block_cols_A, MPI_DOUBLE, c, row_comm);
    MPI_Bcast(B_row_host, block_rows_B * block_cols_B, MPI_DOUBLE, r, col_comm);

    CUDA_CHECK(cudaMemcpy(d_A_col, A_col_host, block_rows_A * block_cols_A * sizeof(double), cudaMemcpyHostToDevice), rank);
    CUDA_CHECK(cudaMemcpy(d_B_row, B_row_host, block_rows_B * block_cols_B * sizeof(double), cudaMemcpyHostToDevice), rank);

    dim3 dim_block(tile_width, tile_width, 1);
    dim3 dim_grid(ceil(block_cols_B / (float)tile_width), ceil(block_rows_A / (float)tile_width), 1);
    int shared_mem_size = 2 * tile_width * tile_width * sizeof(double);

    matrix_mul_kernel<<<dim_grid, dim_block, shared_mem_size>>>(d_A_col, d_B_row, d_C_block, block_rows_A, block_cols_B, block_cols_A);

    CUDA_CHECK(cudaGetLastError(), rank);
    CUDA_CHECK(cudaDeviceSynchronize(), rank);
  }

  CUDA_CHECK(cudaMemcpy(C_host_block, d_C_block, block_rows_A * block_cols_B * sizeof(double), cudaMemcpyDeviceToHost), rank);

  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);

  free(A_col_host);
  free(B_row_host);

  CUDA_CHECK(cudaFree(d_A_col), rank);
  CUDA_CHECK(cudaFree(d_B_row), rank);
  CUDA_CHECK(cudaFree(d_C_block), rank);
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