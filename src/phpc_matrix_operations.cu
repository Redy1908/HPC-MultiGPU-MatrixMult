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

void phpc_gemm_summa_cuda(MPI_Comm grid_comm, double *A, double *B, double *C, int lda, int ldb, int ldc, int Nglob, int Mglob, int Pglob, dim3 dim_block, dim3 dim_grid, int shared_mem_size) {
  int dims[2], periods[2], coords[2];
  int k_loop, c_col_rank, r_row_rank, K2_lcm;
  dim3 actual_dim_grid;

  int h_block_rows_A, block_rows_B_panel;
  int block_cols_A_panel, h_block_cols_B;

  double *h_A_tile_start, *h_B_tile_start;
  double *d_A_tile, *d_B_tile, *d_C_tile;

  MPI_Comm row_comm, col_comm;

  int remain_dims_row[2] = {0, 1};
  int remain_dims_col[2] = {1, 0};

  MPI_Cart_get(grid_comm, 2, dims, periods, coords);

  K2_lcm = find_lcm(dims[0], dims[1]);

  h_block_rows_A = Nglob / dims[0];
  block_cols_A_panel = Mglob / K2_lcm;
  block_rows_B_panel = Mglob / K2_lcm;
  h_block_cols_B = Pglob / dims[1];

  cudaMalloc((void **)&d_A_tile, h_block_rows_A * block_cols_A_panel * sizeof(double));
  cudaMalloc((void **)&d_B_tile, block_rows_B_panel * h_block_cols_B * sizeof(double));
  cudaMalloc((void **)&d_C_tile, h_block_rows_A * h_block_cols_B * sizeof(double));

  cudaMemset(d_C_tile, 0, h_block_rows_A * h_block_cols_B * sizeof(double));

  MPI_Cart_sub(grid_comm, remain_dims_row, &row_comm);
  MPI_Cart_sub(grid_comm, remain_dims_col, &col_comm);

  h_A_tile_start = A;
  h_B_tile_start = B;

  for (k_loop = 0; k_loop < K2_lcm; k_loop++) {
    c_col_rank = k_loop % dims[1];
    r_row_rank = k_loop % dims[0];

    if (coords[1] == c_col_rank) {
      cudaMemcpy2D(d_A_tile,
                   block_cols_A_panel * sizeof(double),
                   h_A_tile_start,
                   lda * sizeof(double),
                   block_cols_A_panel * sizeof(double),
                   h_block_rows_A,
                   cudaMemcpyHostToDevice);
      h_A_tile_start += block_cols_A_panel;
    }

    if (coords[0] == r_row_rank) {
      cudaMemcpy2D(d_B_tile,
                   h_block_cols_B * sizeof(double),
                   h_B_tile_start,
                   ldb * sizeof(double),
                   h_block_cols_B * sizeof(double),
                   block_rows_B_panel,
                   cudaMemcpyHostToDevice);
      h_B_tile_start += block_rows_B_panel * ldb;
    }

    MPI_Bcast(d_A_tile, h_block_rows_A * block_cols_A_panel, MPI_DOUBLE, c_col_rank, row_comm);
    MPI_Bcast(d_B_tile, block_rows_B_panel * h_block_cols_B, MPI_DOUBLE, r_row_rank, col_comm);

    if (dim_grid.x == 0 && dim_grid.y == 0) {
      actual_dim_grid = dim3(ceil((double)h_block_cols_B / dim_block.x), ceil((double)h_block_rows_A / dim_block.y));
    } else {
      actual_dim_grid = dim_grid;
      actual_dim_grid.z = 1;
    }

    gemm_kernel<<<actual_dim_grid, dim_block, shared_mem_size>>>(
        d_A_tile, d_B_tile, d_C_tile,
        h_block_rows_A, h_block_cols_B, block_cols_A_panel);

    cudaDeviceSynchronize();
  }

  cudaMemcpy2D(C,
               ldc * sizeof(double),
               d_C_tile,
               h_block_cols_B * sizeof(double),
               h_block_cols_B * sizeof(double),
               h_block_rows_A,
               cudaMemcpyDeviceToHost);

  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);

  cudaFree(d_A_tile);
  cudaFree(d_B_tile);
  cudaFree(d_C_tile);
}