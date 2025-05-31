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
  int k, is_col, is_row, lcm;

  int local_A_rows;
  int panel_K_dim;
  int local_B_cols;

  double *d_A_panel, *d_B_panel, *d_C_local_block;

  dim3 actual_dim_grid;

  MPI_Comm row_comm, col_comm;

  int remain_dims_row[2] = {0, 1};
  int remain_dims_col[2] = {1, 0};

  MPI_Cart_get(grid_comm, 2, dims, periods, coords);

  lcm = find_lcm(dims[0], dims[1]);

  local_A_rows = Nglob / dims[0];
  panel_K_dim = Mglob / lcm;
  local_B_cols = Pglob / dims[1];

  cudaMalloc((void **)&d_A_panel, local_A_rows * panel_K_dim * sizeof(double));
  cudaMalloc((void **)&d_B_panel, panel_K_dim * local_B_cols * sizeof(double));
  cudaMalloc((void **)&d_C_local_block, local_A_rows * local_B_cols * sizeof(double));

  cudaMemset(d_C_local_block, 0, local_A_rows * local_B_cols * sizeof(double));

  MPI_Cart_sub(grid_comm, remain_dims_row, &row_comm);
  MPI_Cart_sub(grid_comm, remain_dims_col, &col_comm);

  double *current_A_panel_host_start = A;
  double *current_B_panel_host_start = B;

  for (k = 0; k < lcm; k++) {
    is_col = k % dims[1];
    is_row = k % dims[0];

    if (coords[1] == is_col) {
      cudaMemcpy2D(d_A_panel,
                   panel_K_dim * sizeof(double),
                   current_A_panel_host_start,
                   lda * sizeof(double),
                   panel_K_dim * sizeof(double),
                   local_A_rows,
                   cudaMemcpyHostToDevice);
      current_A_panel_host_start += panel_K_dim;
    }

    if (coords[0] == is_row) {
      cudaMemcpy2D(d_B_panel,
                   local_B_cols * sizeof(double),
                   current_B_panel_host_start,
                   ldb * sizeof(double),
                   local_B_cols * sizeof(double),
                   panel_K_dim,
                   cudaMemcpyHostToDevice);
      current_B_panel_host_start += panel_K_dim * ldb;
    }

    MPI_Bcast(d_A_panel, local_A_rows * panel_K_dim, MPI_DOUBLE, is_col, row_comm);
    MPI_Bcast(d_B_panel, panel_K_dim * local_B_cols, MPI_DOUBLE, is_row, col_comm);

    if (dim_grid.x == 0 && dim_grid.y == 0) {
      actual_dim_grid.x = (unsigned int)ceil((double)local_B_cols / dim_block.x);
      actual_dim_grid.y = (unsigned int)ceil((double)local_A_rows / dim_block.y);
      actual_dim_grid.z = 1;
    } else {
      actual_dim_grid = dim_grid;
    }

    gemm_kernel<<<actual_dim_grid, dim_block, shared_mem_size>>>(d_A_panel, d_B_panel, d_C_local_block, local_A_rows, local_B_cols, panel_K_dim);

    cudaDeviceSynchronize();
  }

  cudaMemcpy2D(C,
               (size_t)ldc * sizeof(double),
               d_C_local_block,
               (size_t)local_B_cols * sizeof(double),
               (size_t)local_B_cols * sizeof(double),
               (size_t)local_A_rows,
               cudaMemcpyDeviceToHost);

  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);

  cudaFree(d_A_panel);
  cudaFree(d_B_panel);
  cudaFree(d_C_local_block);
}