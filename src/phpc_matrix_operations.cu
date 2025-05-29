#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "phpc_matrix_operations.cuh"
#include "utils.cuh"

#define IDX(row, col, num_cols) ((row) * (num_cols) + (col))

__global__ void gemm_kernel(double *A, double *B, double *C, int M, int N, int K) {
  extern __shared__ double shared_mem[];

  int tile_width = blockDim.x;

  double *s_A = (double *)shared_mem;
  double *s_B = (double *)shared_mem + tile_width * tile_width;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int num_tiles_along_N = (int)ceil((double)N / tile_width);
  int num_tiles_along_M = (int)ceil((double)M / tile_width);
  int total_output_tiles = num_tiles_along_M * num_tiles_along_N;

  int block_id_1d = blockIdx.y * gridDim.x + blockIdx.x;
  int total_launched_blocks = gridDim.x * gridDim.y;

  // Grid-stride loop: ogni blocco itera sulle tile di C che gli sono assegnate
  for (int current_tile_1d_idx = block_id_1d; current_tile_1d_idx < total_output_tiles; current_tile_1d_idx += total_launched_blocks) {
    int by_tile = current_tile_1d_idx / num_tiles_along_N;
    int bx_tile = current_tile_1d_idx % num_tiles_along_N;

    int C_tile_row_base = by_tile * tile_width;
    int C_tile_col_base = bx_tile * tile_width;

    int global_row_C = C_tile_row_base + ty;
    int global_col_C = C_tile_col_base + tx;

    double c_value = 0.0;

    int num_phases = (int)ceil((double)K / tile_width);
    for (int phase = 0; phase < num_phases; ++phase) {
      if ((global_row_C < M) && (phase * tile_width + tx) < K)
        s_A[ty * tile_width + tx] = A[global_row_C * K + phase * tile_width + tx];
      else
        s_A[ty * tile_width + tx] = 0.0;

      if ((phase * tile_width + ty) < K && (global_col_C < N))
        s_B[ty * tile_width + tx] = B[(phase * tile_width + ty) * N + global_col_C];
      else
        s_B[ty * tile_width + tx] = 0.0;

      __syncthreads();

      for (int k_tile = 0; k_tile < tile_width; ++k_tile) {
        if (phase * tile_width + k_tile < K) {
          c_value += s_A[ty * tile_width + k_tile] * s_B[k_tile * tile_width + tx];
        }
      }
      __syncthreads();
    }

    if ((global_row_C < M) && (global_col_C < N))
      C[global_row_C * N + global_col_C] += c_value;
  }
}

int phpc_gemm_sequential(const double *A, const double *B, double *C, unsigned int m, unsigned int k, unsigned int n) {
  for (unsigned int i = 0; i < m; ++i)
    for (unsigned int j = 0; j < n; ++j)
      for (unsigned int l = 0; l < k; ++l)
        C[IDX(i, j, n)] += A[IDX(i, l, k)] * B[IDX(l, j, n)];

  return 0;
}

int phpc_gemm_cuda(const double *A, const double *B, double *C, unsigned int m, unsigned int k, unsigned int n, dim2 grid_size, unsigned int block_width) {
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
  dim3 kernel_block_size(block_width, block_width, 1);
  gemm_kernel<<<kernel_grid_size, kernel_block_size, shared_mem_size>>>(A_dev, B_dev, C_dev, m, n, k);
  cudaDeviceSynchronize();

  cudaMemcpy(C, C_dev, m * n * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(A_dev);
  cudaFree(B_dev);
  cudaFree(C_dev);

  return 0;
}

int phpc_gemm_cublas(const double *A, const double *B, double *C, int m, int k, int n) {
  if (m < 0 || k < 0 || n < 0)
    return 1;

  double *A_dev, *B_dev, *C_dev;

  if (cudaMalloc(&A_dev, m * k * sizeof(double)) != cudaSuccess)
    return 1;

  if (cudaMalloc(&B_dev, k * n * sizeof(double)) != cudaSuccess) {
    cudaFree(A_dev);
    return 1;
  }

  if (cudaMalloc(&C_dev, m * n * sizeof(double)) != cudaSuccess) {
    cudaFree(B_dev);
    cudaFree(A_dev);
    return 1;
  }

  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    cudaFree(C_dev);
    cudaFree(B_dev);
    cudaFree(A_dev);
    return 1;
  }

  cudaMemcpy(A_dev, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(B_dev, B, k * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(C_dev, C, m * n * sizeof(double), cudaMemcpyHostToDevice);

  const double alpha = 1;
  const double beta = 1;

  /* https://stackoverflow.com/a/56064726/17731255 */
  if (cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B_dev, n, A_dev, k, &beta, C_dev, n) != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(handle);
    cudaFree(C_dev);
    cudaFree(B_dev);
    cudaFree(A_dev);
    return 1;
  }

  cudaDeviceSynchronize();
  cudaMemcpy(C, C_dev, m * n * sizeof(double), cudaMemcpyDeviceToHost);

  cublasDestroy(handle);
  cudaFree(C_dev);
  cudaFree(B_dev);
  cudaFree(A_dev);

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

int phpc_gemm_summa_cuda(const MPI_Comm grid_comm, double *A, double *B, double *C, unsigned int m, unsigned int k, unsigned int n, dim2 grid_size, unsigned int block_width) {
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
    free(buffer_b);
    free(buffer_a);
    return 1;
  }

  double *A_dev, *B_dev, *C_dev;

  if (cudaMalloc(&A_dev, sub_m * max_sub_k * sizeof(double)) != cudaSuccess) {
    cudaFree(A_dev);
    free(buffer_b);
    free(buffer_a);
    return 1;
  }

  if (cudaMalloc(&B_dev, max_sub_k * sub_n * sizeof(double)) != cudaSuccess) {
    cudaFree(B_dev);
    cudaFree(A_dev);
    free(buffer_b);
    free(buffer_a);
    return 1;
  }

  if (cudaMalloc(&C_dev, sub_m * sub_n * sizeof(double)) != cudaSuccess) {
    cudaFree(C_dev);
    cudaFree(B_dev);
    cudaFree(A_dev);
    free(buffer_b);
    free(buffer_a);
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
    uint shared_mem_size = 2 * block_width * block_width * sizeof(double);
    dim3 kernel_grid_size(grid_size.x, grid_size.y, 1);
    dim3 kernel_block_size(block_width, block_width, 1);
    gemm_kernel<<<kernel_grid_size, kernel_block_size, shared_mem_size>>>(A_dev, B_dev, C_dev, sub_m, sub_n, sub_k);
    cudaDeviceSynchronize();
  }

  /* copy the final result from the GPU to the CPU */
  cudaMemcpy(C, C_dev, sub_m * sub_n * sizeof(double), cudaMemcpyDeviceToHost);

  /* cleanup */
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
  cudaFree(C_dev);
  cudaFree(B_dev);
  cudaFree(A_dev);
  free(buffer_b);
  free(buffer_a);

  return 0;
}

int phpc_gemm_summa_cublas(const MPI_Comm grid_comm, double *A, double *B, double *C, int m, int k, int n) {
  if (m < 0 || k < 0 || n < 0)
    return 1;

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
    free(buffer_b);
    free(buffer_a);
    return 1;
  }

  double *A_dev, *B_dev, *C_dev;

  if (cudaMalloc(&A_dev, sub_m * max_sub_k * sizeof(double)) != cudaSuccess) {
    cudaFree(A_dev);
    free(buffer_b);
    free(buffer_a);
    return 1;
  }

  if (cudaMalloc(&B_dev, max_sub_k * sub_n * sizeof(double)) != cudaSuccess) {
    cudaFree(B_dev);
    cudaFree(A_dev);
    free(buffer_b);
    free(buffer_a);
    return 1;
  }

  if (cudaMalloc(&C_dev, sub_m * sub_n * sizeof(double)) != cudaSuccess) {
    cudaFree(C_dev);
    cudaFree(B_dev);
    cudaFree(A_dev);
    free(buffer_b);
    free(buffer_a);
    return 1;
  }

  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    cudaFree(C_dev);
    cudaFree(B_dev);
    cudaFree(A_dev);
    free(buffer_b);
    free(buffer_a);
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

  const double alpha = 1;
  const double beta = 1;

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
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_n, sub_m, sub_k, &alpha, block_b, sub_n, block_a, sub_k, &beta, C_dev, sub_n);
  }

  /* copy the final result from the GPU to the CPU */
  cudaMemcpy(C, C_dev, sub_m * sub_n * sizeof(double), cudaMemcpyDeviceToHost);

  /* cleanup */
  cublasDestroy(handle);
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
  cudaFree(C_dev);
  cudaFree(B_dev);
  cudaFree(A_dev);
  free(buffer_b);
  free(buffer_a);

  return 0;
}

#undef IDX
