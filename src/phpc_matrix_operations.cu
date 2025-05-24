#include <cuda_runtime.h>
#include <mpi/mpi.h>
#include <stdlib.h>

#include "phpc_matrix_operations.cuh"

#define IDX(row, col, num_cols) ((row) * (num_cols) + (col))

int find_lcm(int a, int b) {
  int q, r;
  int x = a;
  int y = b;

  while (y != 0) {
    q = x / y;
    r = x - q * y;
    x = y;
    y = r;
  }

  return a * b / x;
}

__global__ void gemm_kernel(double *A, double *B, double *C, unsigned int m, unsigned int n, unsigned int k) {
  extern __shared__ double shared_mem[];

  // int tile_width = blockDim.x;

  double *s_A = shared_mem;
  double *s_B = shared_mem + blockDim.x * blockDim.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  double c_value = 0.0;
  unsigned int phases = k / blockDim.x + (k % blockDim.x > 0);

  for (int phase = 0; phase < phases; phase++) {
    if ((row < m) && (phase * blockDim.x + tx) < k)
      s_A[ty * blockDim.x + tx] = A[row * k + phase * blockDim.x + tx];
    else
      s_A[ty * blockDim.x + tx] = 0.0;

    if ((phase * blockDim.y + ty) < k && (col < n))
      s_B[ty * blockDim.x + tx] = B[(phase * blockDim.y + ty) * n + col];
    else
      s_B[ty * blockDim.x + tx] = 0.0;

    __syncthreads();

    for (int k = 0; k < blockDim.x; ++k) {
      c_value += s_A[ty * blockDim.x + k] * s_B[k * blockDim.y + tx];
    }
    __syncthreads();
  }

  if (row < m && col < n)
    C[row * n + col] += c_value;
}

int phpc_gemm_sequential(const double *A, const double *B, double *C, unsigned int m, unsigned int k, unsigned int n) {
  for (unsigned int i = 0; i < m; ++i)
    for (unsigned int j = 0; j < n; ++j)
      for (unsigned int l = 0; l < k; ++l)
        C[IDX(i, j, n)] += A[IDX(i, l, k)] * B[IDX(l, j, n)];

  return 0;
}

int phpc_gemm_cuda(const double *A, const double *B, double *C, unsigned int m, unsigned int k, unsigned int n, dim2 grid_size, dim2 block_size) {
  double *A_dev, *B_dev, *C_dev;
  cudaMalloc(&A_dev, m * k * sizeof(double));
  cudaMalloc(&B_dev, k * n * sizeof(double));
  cudaMalloc(&C_dev, m * n * sizeof(double));

  if (A_dev == NULL || B_dev == NULL || C_dev == NULL) {
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    return 1;
  }

  uint shared_mem_size = (m * k + k * n) * sizeof(double);
  cudaMemcpy(A_dev, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(B_dev, B, k * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(C_dev, C, m * n * sizeof(double), cudaMemcpyHostToDevice);

  dim3 kernel_grid_size(grid_size.x, grid_size.y, 1);
  dim3 kernel_block_size(block_size.x, block_size.y, 1);
  gemm_kernel<<<kernel_grid_size, kernel_block_size, shared_mem_size>>>(A_dev, B_dev, C_dev, m, n, k);
  cudaDeviceSynchronize();

  cudaMemcpy(C, C_dev, m * n * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(A_dev);
  cudaFree(B_dev);
  cudaFree(C_dev);

  return 0;
}

int phpc_gemm_summa_sequential(const MPI_Comm grid_comm, double *A, double *B, double *C, unsigned int m, unsigned int k, unsigned int n) {
  int rank;
  int dims[2], periods[2], coords[2];

  MPI_Comm_rank(grid_comm, &rank);
  MPI_Cart_get(grid_comm, 2, dims, periods, coords);

  int K2 = find_lcm(dims[0], dims[1]);

  uint sub_m = m / dims[0] + (coords[0] < m % dims[0]); /* numbers of row for the A sub-block */
  uint sub_n = n / dims[1] + (coords[1] < n % dims[1]); /* numbers of columns for the B sub-block */
  uint max_sub_k = k / K2 + (k % K2 > 0);

  double *buffer_a = (double *)malloc(sub_m * max_sub_k * sizeof(double));
  double *buffer_b = (double *)malloc(max_sub_k * sub_n * sizeof(double));

  if (buffer_a == NULL || buffer_b == NULL) {
    free(buffer_a);
    free(buffer_b);
    return 1;
  }

  /* create communicators along rows and columns */
  int remain_dims_row[2] = {0, 1};
  int remain_dims_col[2] = {1, 0};
  MPI_Comm row_comm, col_comm;
  MPI_Cart_sub(grid_comm, remain_dims_row, &row_comm);
  MPI_Cart_sub(grid_comm, remain_dims_col, &col_comm);

  for (uint i = 0; i < K2; i++) {
    uint r = i % dims[0];
    uint c = i % dims[1];

    /* calculate the k of the sending process */
    uint sub_k = k / K2 + (i < k % K2);

    /**
     * NOTE 2025-05-22 (Salvatore)
     * the following assumes that blocks assigned to the process are stored contiguosly one after the other and NOT interleaved
     * wrong format: row 0 of block 0, row 0 of block 1, row 1 of block 0, row 1 of block 1, ...
     * correct format: row 0 of block 0, row 1 of block 0, row n of block 0, row 0 of block 1, ...
     */

    double *block_a, *block_b;

    if (coords[1] == c) {
      block_a = A;        /* we are sending the block */
      A += sub_m * sub_k; /* we may have to send again in the future, skip the pointer to the start of the other block */
    } else {
      block_a = buffer_a; /* we are receiving, prepare the buffer */
    }

    if (coords[0] == r) {
      block_b = B;        /* we are sending the block */
      B += sub_k * sub_n; /* we may have to send again in the future, skip the pointer to the start of the other block */
    } else {
      block_b = buffer_b; /* we are receiving, prepare the buffer */
    }

    /* a process broadcasts one of its blocks of A on all the other processes in the same row */
    MPI_Bcast(block_a, sub_m * sub_k, MPI_DOUBLE, c, row_comm);

    /* a process broadcasts one of its blocks of B on all the other processes in the same column */
    MPI_Bcast(block_b, sub_k * sub_n, MPI_DOUBLE, r, col_comm);

    /* compute the submatrices product */
    phpc_gemm_sequential(block_a, block_b, C, sub_m, sub_k, sub_n);
  }

  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);

  free(buffer_a);
  free(buffer_b);

  return 0;
}

int phpc_gemm_summa_cuda(const MPI_Comm grid_comm, double *A, double *B, double *C, unsigned int m, unsigned int k, unsigned int n, dim2 grid_size, dim2 block_size) {
  int rank;
  int dims[2], periods[2], coords[2];

  MPI_Comm_rank(grid_comm, &rank);
  MPI_Cart_get(grid_comm, 2, dims, periods, coords);

  int K2 = find_lcm(dims[0], dims[1]);

  uint sub_m = m / dims[0] + (coords[0] < m % dims[0]); /* numbers of row for the A sub-block */
  uint sub_n = n / dims[1] + (coords[1] < n % dims[1]); /* numbers of columns for the B sub-block */
  uint max_sub_k = k / K2 + (k % K2 > 0);

  double *buffer_a = (double *)malloc(sub_m * max_sub_k * sizeof(double));
  double *buffer_b = (double *)malloc(max_sub_k * sub_n * sizeof(double));

  double *A_dev, *B_dev, *C_dev;
  cudaMalloc(&A_dev, sub_m * max_sub_k * sizeof(double));
  cudaMalloc(&B_dev, max_sub_k * sub_n * sizeof(double));
  cudaMalloc(&C_dev, sub_m * sub_n * sizeof(double));

  if (buffer_a == NULL || buffer_b == NULL || A_dev == NULL || B_dev == NULL || C_dev == NULL) {
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    free(buffer_a);
    free(buffer_b);
    return 1;
  }

  /* copy result matrix to GPU */
  cudaMemcpy(C_dev, C, m * n * sizeof(double), cudaMemcpyHostToDevice);

  /* create communicators along rows and columns */
  int remain_dims_row[2] = {0, 1};
  int remain_dims_col[2] = {1, 0};
  MPI_Comm row_comm, col_comm;
  MPI_Cart_sub(grid_comm, remain_dims_row, &row_comm);
  MPI_Cart_sub(grid_comm, remain_dims_col, &col_comm);

  for (uint i = 0; i < K2; i++) {
    uint r = i % dims[0];
    uint c = i % dims[1];

    /* calculate the k of the sending process */
    uint sub_k = k / K2 + (i < k % K2);

    /**
     * NOTE 2025-05-22 (Salvatore)
     * the following assumes that blocks assigned to the process are stored contiguosly one after the other and NOT interleaved
     * wrong format: row 0 of block 0, row 0 of block 1, row 1 of block 0, row 1 of block 1, ...
     * correct format: row 0 of block 0, row 1 of block 0, row n of block 0, row 0 of block 1, ...
     */

    double *block_a, *block_b;

    if (coords[1] == c) {
      block_a = A;        /* we are sending the block */
      A += sub_m * sub_k; /* we may have to send again in the future, skip the pointer to the start of the other block */
    } else {
      block_a = buffer_a; /* we are receiving, prepare the buffer */
    }

    if (coords[0] == r) {
      block_b = B;        /* we are sending the block */
      B += sub_k * sub_n; /* we may have to send again in the future, skip the pointer to the start of the other block */
    } else {
      block_b = buffer_b; /* we are receiving, prepare the buffer */
    }

    /* a process broadcasts one of its blocks of A on all the other processes in the same row */
    MPI_Bcast(block_a, sub_m * sub_k, MPI_DOUBLE, c, row_comm);

    /* a process broadcasts one of its blocks of B on all the other processes in the same column */
    MPI_Bcast(block_b, sub_k * sub_n, MPI_DOUBLE, r, col_comm);

    /* copy blocks from CPU to GPU */
    cudaMemcpy(A_dev, block_a, sub_m * sub_k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, block_b, sub_k * sub_n * sizeof(double), cudaMemcpyHostToDevice);

    /* compute product on the GPU */
    uint shared_mem_size = (sub_m * sub_n * 2) * sizeof(double);
    dim3 kernel_grid_size(grid_size.x, grid_size.y, 1);
    dim3 kernel_block_size(block_size.x, block_size.y, 1);
    gemm_kernel<<<kernel_grid_size, kernel_block_size, shared_mem_size>>>(A_dev, B_dev, C_dev, sub_m, sub_n, sub_k);
    cudaDeviceSynchronize();
  }

  /* copy the final result from the GPU to the CPU */
  cudaMemcpy(C, C_dev, sub_m * sub_n * sizeof(double), cudaMemcpyDeviceToHost);

  /* cleanup */
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);

  cudaFree(A_dev);
  cudaFree(B_dev);
  cudaFree(C_dev);

  free(buffer_a);
  free(buffer_b);

  return 0;
}

#undef IDX
