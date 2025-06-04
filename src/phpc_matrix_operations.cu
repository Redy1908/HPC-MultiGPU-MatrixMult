#include <assert.h>
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

int phpc_gemm_cuda(const double *a, int lda, const double *b, int ldb, double *c, int ldc, int width, int gpu_count, int grid_width, int grid_height, int block_width) {
  int device_count;
  cudaGetDeviceCount(&device_count);

  assert(width > 0);
  assert(gpu_count > 0 && gpu_count <= device_count);
  assert(grid_width > 0 && grid_height > 0);
  assert(block_width * block_width <= 1024);
  assert(width % gpu_count == 0);

  /**
   * Matrix A: each gpu has a "row"
   *  ________________________
   * |         GPU 0         |
   * -------------------------
   * |          ...          |
   * -------------------------
   * |         GPU N         |
   * -------------------------
   *
   * Matrix B: each gpu has a "column"
   *  ________________________
   * |       |       |       |
   * |       |       |       |
   * | GPU 0 |  ...  | GPU N |
   * |       |       |       |
   * |       |       |       |
   * -------------------------
   */

  int m = width / gpu_count;
  int k = width;

  double **dev_buffers_a = (double **)malloc(gpu_count * sizeof(double *));
  double **dev_buffers_b = (double **)malloc(gpu_count * sizeof(double *));
  double **dev_buffers_c = (double **)malloc(gpu_count * sizeof(double *));
  cudaStream_t *streams = (cudaStream_t *)malloc(gpu_count * sizeof(cudaStream_t));

  assert(dev_buffers_a != NULL);
  assert(dev_buffers_b != NULL);
  assert(dev_buffers_c != NULL);
  assert(streams != NULL);

  for (int i = 0; i < gpu_count; i++) {
    cudaSetDevice(i);
    cudaMalloc(&(dev_buffers_a[i]), m * k * sizeof(double));
    cudaMalloc(&(dev_buffers_b[i]), m * k * sizeof(double));
    cudaMalloc(&(dev_buffers_c[i]), m * k * sizeof(double));
    cudaStreamCreate(&(streams[i]));

    /* copy column of B*/
    cudaMemcpy2DAsync(dev_buffers_b[i], m * sizeof(double), b + m * i, ldb * sizeof(double), m * sizeof(double), width, cudaMemcpyHostToDevice, streams[i]);

    /* copy column of C */
    cudaMemcpy2DAsync(dev_buffers_c[i], m * sizeof(double), c + m * i, ldc * sizeof(double), m * sizeof(double), width, cudaMemcpyHostToDevice, streams[i]);
  }

  dim3 grid_size(grid_width, grid_height, 1);
  dim3 block_size(block_width, block_width, 1);
  int shared_memory_size = 2 * block_width * block_width * sizeof(double);

  for (int i = 0; i < gpu_count; i++) {
    for (size_t j = 0; j < gpu_count; j++) {
      cudaSetDevice(j);

      /* copy row j % gpu_count of A */
      int row = (j + i) % gpu_count;
      cudaMemcpyAsync(dev_buffers_a[j], a + m * k * row, m * k * sizeof(double), cudaMemcpyHostToDevice, streams[j]);

      int dev_offset_c = m * m * row;
      gemm_kernel<<<grid_size, block_size, shared_memory_size, streams[j]>>>(dev_buffers_a[j], dev_buffers_b[j], dev_buffers_c[j] + dev_offset_c, m, m, k);
    }
  }

  /**
   * Gather results: each GPU has a column of the resulting matrix C
   *  ________________________
   * |       |       |       |
   * |       |       |       |
   * | GPU 0 |  ...  | GPU i |
   * |       |       |       |
   * |       |       |       |
   * -------------------------
   */
  for (int i = 0; i < gpu_count; i++) {
    cudaSetDevice(i);
    cudaMemcpy2D(c + m * i, width * sizeof(double), dev_buffers_c[i], m * sizeof(double), m * sizeof(double), width, cudaMemcpyDeviceToHost);

    cudaFree(&(dev_buffers_c[i]));
    cudaFree(&(dev_buffers_b[i]));
    cudaFree(&(dev_buffers_a[i]));
    cudaStreamDestroy(streams[i]);
  }

  free(streams);
  free(dev_buffers_c);
  free(dev_buffers_b);
  free(dev_buffers_a);

  return 0;
}

void phpc_gemm_summa_cuda(MPI_Comm grid_comm, const double *A, const double *B, double *C, int lda, int ldb, int ldc, int matrices_width, int gpu_count, int grid_width, int grid_height, int block_width) {
  assert(lda > 0);
  assert(ldb > 0);
  assert(ldc > 0);

  MPI_Comm row_comm, col_comm;
  int dims[2], periods[2], coords[2];
  int remain_dims_row[2] = {0, 1};
  int remain_dims_col[2] = {1, 0};
  MPI_Cart_get(grid_comm, 2, dims, periods, coords);
  MPI_Cart_sub(grid_comm, remain_dims_row, &row_comm);
  MPI_Cart_sub(grid_comm, remain_dims_col, &col_comm);

  int lcm = find_lcm(dims[0], dims[1]);
  int local_A_rows = matrices_width / dims[0];
  int panel_K_dim = matrices_width / lcm;
  int local_B_cols = matrices_width / dims[1];

  assert(matrices_width % dims[0] == 0);
  assert(matrices_width % lcm == 0);

  /* compute optimal size */
  if (grid_width == 0 && grid_height == 0) {
    grid_width = local_B_cols / block_width + (local_B_cols % block_width > 0);
    grid_height = local_A_rows / block_width + (local_A_rows % block_width > 0);
  }

  double *buffer_a = (double *)malloc(local_A_rows * panel_K_dim * sizeof(double));
  double *buffer_b = (double *)malloc(panel_K_dim * local_B_cols * sizeof(double));

  assert(buffer_a != NULL);
  assert(buffer_b != NULL);

  for (int k = 0; k < lcm; k++) {
    int sender_column = k % dims[1];
    int sender_row = k % dims[0];

    const double *block_a, *block_b;

    if (coords[1] == sender_column) {
      block_a = A;                     /* we are sending the block */
      A += local_A_rows * panel_K_dim; /* we may have to send again in the future, skip the pointer to the start of the other block */
    } else {
      block_a = buffer_a; /* we are receiving, prepare the buffer */
    }

    if (coords[0] == sender_row) {
      block_b = B;                     /* we are sending the block */
      B += panel_K_dim * local_B_cols; /* we may have to send again in the future, skip the pointer to the start of the other block */
    } else {
      block_b = buffer_b; /* we are receiving, prepare the buffer */
    }

    MPI_Bcast((void *)block_a, local_A_rows * panel_K_dim, MPI_DOUBLE, sender_column, row_comm);
    MPI_Bcast((void *)block_b, panel_K_dim * local_B_cols, MPI_DOUBLE, sender_row, col_comm);

    phpc_gemm_cuda(block_a, lda, block_b, ldb, C, ldc, matrices_width, gpu_count, grid_width, grid_height, block_width);
  }

  // TODO: gather matrices

  free(buffer_b);
  free(buffer_a);

  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
}
