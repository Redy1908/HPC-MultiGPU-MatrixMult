#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#include "phpc_matrix_operations.cuh"
#include "utils.cuh"

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

void phpc_gemm_summa_cuda(MPI_Comm grid_comm, double *A, double *B, double *C, int ld, int M, int K, int N, dim3 dim_block, dim3 dim_grid, int shared_mem_size) {
  int dims[2], periods[2], coords[2];
  int i, k, c, r, K2;

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

  B_row_host = (double *)malloc(block_rows_B * block_cols_B * sizeof(double));

  cudaMalloc((void **)&d_A_col, block_rows_A * block_cols_A * sizeof(double));
  cudaMalloc((void **)&d_B_row, block_rows_B * block_cols_B * sizeof(double));
  cudaMalloc((void **)&d_C_block, block_rows_A * block_cols_B * sizeof(double));

  cudaMemset(d_C_block, 0, block_rows_A * block_cols_B * sizeof(double));

  MPI_Cart_sub(grid_comm, remain_dims_row, &row_comm);
  MPI_Cart_sub(grid_comm, remain_dims_col, &col_comm);

  A_start = A;
  B_start = B;

  for (k = 0; k < K2; k++) {
    c = k % dims[1];
    r = k % dims[0];

    if (coords[1] == c) {
      for (i = 0; i < block_rows_A; i++) {
        memcpy(&A_col_host[i * block_cols_A], &A_start[i * ld], block_cols_A * sizeof(double));
      }
      A_start += block_cols_A;
    }

    if (coords[0] == r) {
      for (i = 0; i < block_rows_B; i++) {
        memcpy(&B_row_host[i * block_cols_B], &B_start[i * ld], block_cols_B * sizeof(double));
      }
      B_start += block_rows_B * ld;
    }

    MPI_Bcast(A_col_host, block_rows_A * block_cols_A, MPI_DOUBLE, c, row_comm);
    MPI_Bcast(B_row_host, block_rows_B * block_cols_B, MPI_DOUBLE, r, col_comm);

    cudaMemcpy(d_A_col, A_col_host, block_rows_A * block_cols_A * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_row, B_row_host, block_rows_B * block_cols_B * sizeof(double), cudaMemcpyHostToDevice);

    dim_block.z = 1;
    dim_grid.z = 1;
    gemm_kernel<<<dim_grid, dim_block, shared_mem_size>>>(d_A_col, d_B_row, d_C_block, block_rows_A, block_cols_B, block_cols_A);

    cudaGetLastError();
    cudaDeviceSynchronize();
  }

  cudaMemcpy(C, d_C_block, block_rows_A * block_cols_B * sizeof(double), cudaMemcpyDeviceToHost);

  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);

  free(A_col_host);
  free(B_row_host);

  cudaFree(d_A_col);
  cudaFree(d_B_row);
  cudaFree(d_C_block);
}