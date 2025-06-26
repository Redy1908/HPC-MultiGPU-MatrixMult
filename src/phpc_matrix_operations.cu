#include <assert.h>
#include <cublasXt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#include "phpc_matrix_operations.cuh"
#include "utils.cuh"

typedef void (*gemm_t)(const double *a, int lda, const double *b, int ldb, double *c, int ldc, int m, int k, int n, int gpu_count, int grid_width, int grid_height, int block_width);

void phpc_gemm_iterative(const double *A, const double *B, double *C, int N) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        C[i * N + k] += A[i * N + j] * B[j * N + k];
      }
    }
  }
}

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

  // Grid-stride loop: each block processes multiple tiles
  for (int current_tile_1d_idx = block_id_1d; current_tile_1d_idx < total_output_tiles; current_tile_1d_idx += total_launched_blocks) {
    int by_tile = current_tile_1d_idx / num_tiles_along_N;
    int bx_tile = current_tile_1d_idx % num_tiles_along_N;

    int global_row_C = by_tile * tile_width + ty;
    int global_col_C = bx_tile * tile_width + tx;

    double c_value = 0.0;

    int phases = K / tile_width + (K / tile_width != 0);
    for (int phase = 0; phase < phases; ++phase) {
      if ((global_row_C < M) && (phase * tile_width + tx) < K)
        s_A[ty * tile_width + tx] = A[global_row_C * K + phase * tile_width + tx];
      else
        s_A[ty * tile_width + tx] = 0.0;

      if ((phase * tile_width + ty) < K && (global_col_C < N))
        s_B[ty * tile_width + tx] = B[(phase * tile_width + ty) * N + global_col_C];
      else
        s_B[ty * tile_width + tx] = 0.0;

      __syncthreads();

      for (int k_tile = 0; k_tile < tile_width; ++k_tile)
        c_value += s_A[ty * tile_width + k_tile] * s_B[k_tile * tile_width + tx];

      __syncthreads();
    }

    if ((global_row_C < M) && (global_col_C < N))
      C[global_row_C * N + global_col_C] += c_value;
  }
}

void phpc_gemm_cublas(const double *a, int lda, const double *b, int ldb, double *c, int ldc, int m, int k, int n, int gpu_count, int grid_width, int grid_height, int block_width) {
  int devices[32]; /* checking for 32 devices on a single machine is more than enough */
  cublasXtHandle_t handle;
  double alpha = 1, beta = 1;

  for (size_t i = 0; i < gpu_count; i++)
    devices[i] = i;

  cublasXtCreate(&handle);
  cublasXtDeviceSelect(handle, gpu_count, devices);

  /* note: some subtle math magic to make it work since cublas expects column-major matrices https://stackoverflow.com/a/56064726/17731255 */
  cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, ldb, a, lda, &beta, c, ldc);
  cublasXtDestroy(handle);
}

void phpc_gemm_cuda(const double *a, int lda, const double *b, int ldb, double *c, int ldc, int m, int k, int n, int gpu_count, int grid_width, int grid_height, int block_width) {
  int max_shared_memory_per_block;
  cudaDeviceGetAttribute(&max_shared_memory_per_block, cudaDevAttrMaxSharedMemoryPerBlock, 0);

  int required_shared_memory = 2 * block_width * block_width * sizeof(double);

  if (required_shared_memory > max_shared_memory_per_block)
    printf("Warning: required shared memory exceeds the GPU block limit. This will impact performance.\n");

  /**
   * Matrix A: each gpu copies the entire matrix
   *
   * Matrix B: each gpu has a "column"
   *  ________________________
   * |       |       |       |
   * |       |       |       |
   * | GPU 0 |  ...  | GPU N |
   * |       |       |       |
   * |       |       |       |
   * -------------------------
   *
   * Matrix C: each gpu has a resulting "column"
   *  ________________________
   * |       |       |       |
   * |       |       |       |
   * | GPU 0 |  ...  | GPU i |
   * |       |       |       |
   * |       |       |       |
   * -------------------------
   */

  dim3 grid_size(grid_width, grid_height, 1);
  dim3 block_size(block_width, block_width, 1);

  double **dev_buffers_a = (double **)malloc(gpu_count * sizeof(double *));
  double **dev_buffers_b = (double **)malloc(gpu_count * sizeof(double *));
  double **dev_buffers_c = (double **)malloc(gpu_count * sizeof(double *));
  cudaStream_t *streams = (cudaStream_t *)malloc(gpu_count * sizeof(cudaStream_t));

  for (int gpu = 0; gpu < gpu_count; gpu++) {
    int dev_n = n / gpu_count + (gpu < n % gpu_count);

    cudaSetDevice(gpu);
    cudaStreamCreate(&(streams[gpu]));

    cudaMallocAsync(&(dev_buffers_a[gpu]), m * k * sizeof(double), streams[gpu]);
    cudaMallocAsync(&(dev_buffers_b[gpu]), k * dev_n * sizeof(double), streams[gpu]);
    cudaMallocAsync(&(dev_buffers_c[gpu]), m * dev_n * sizeof(double), streams[gpu]);

    /* copy from host to device */
    cudaMemcpy2DAsync(dev_buffers_a[gpu], k * sizeof(double), a, lda * sizeof(double), k * sizeof(double), m, cudaMemcpyHostToDevice, streams[gpu]);
    cudaMemcpy2DAsync(dev_buffers_b[gpu], dev_n * sizeof(double), b, ldb * sizeof(double), dev_n * sizeof(double), k, cudaMemcpyHostToDevice, streams[gpu]);
    cudaMemcpy2DAsync(dev_buffers_c[gpu], dev_n * sizeof(double), c, ldc * sizeof(double), dev_n * sizeof(double), m, cudaMemcpyHostToDevice, streams[gpu]);

    /* perform computation */
    gemm_kernel<<<grid_size, block_size, required_shared_memory, streams[gpu]>>>(dev_buffers_a[gpu], dev_buffers_b[gpu], dev_buffers_c[gpu], m, dev_n, k);

    /* copy result from device to host */
    cudaMemcpy2DAsync(c, ldc * sizeof(double), dev_buffers_c[gpu], dev_n * sizeof(double), dev_n * sizeof(double), m, cudaMemcpyDeviceToHost, streams[gpu]);

    cudaFreeAsync(dev_buffers_c[gpu], streams[gpu]);
    cudaFreeAsync(dev_buffers_b[gpu], streams[gpu]);
    cudaFreeAsync(dev_buffers_a[gpu], streams[gpu]);

    b += dev_n;
    c += dev_n;
  }

  for (int gpu = 0; gpu < gpu_count; gpu++) {
    cudaSetDevice(gpu);
    cudaStreamSynchronize(streams[gpu]);
    cudaStreamDestroy(streams[gpu]);
  }

  free(streams);
  free(dev_buffers_c);
  free(dev_buffers_b);
  free(dev_buffers_a);
}

int phpc_gemm_summa(gemm_t f, MPI_Comm grid_comm, const double *A, const double *B, double *C, int N, int gpu_count, int grid_width, int grid_height, int block_width) {
  /* get MPI properties */
  int rank, size, dims[2], periods[2], coords[2];
  int remain_dims_row[2] = {0, 1};
  int remain_dims_col[2] = {1, 0};
  MPI_Comm row_comm, col_comm;
  MPI_Comm_rank(grid_comm, &rank);
  MPI_Comm_size(grid_comm, &size);
  MPI_Cart_get(grid_comm, 2, dims, periods, coords);
  MPI_Cart_sub(grid_comm, remain_dims_row, &row_comm);
  MPI_Cart_sub(grid_comm, remain_dims_col, &col_comm);

  int lcm = find_lcm(dims[0], dims[1]);
  int local_A_rows = N / dims[0];
  int panel_K_dim = N / lcm;
  int local_B_cols = N / dims[1];

  /* shift the start of the matrices to the first block actually corresponding to the process */
  A += coords[0] * N * local_A_rows + coords[1] * panel_K_dim;
  B += coords[0] * N * panel_K_dim + coords[1] * local_B_cols;
  double *offset_c = C + coords[0] * N * local_A_rows + coords[1] * local_B_cols;

  /* prepare buffers to receive blocks from other processes */
  double *buffer_a = (double *)malloc(local_A_rows * panel_K_dim * sizeof(double));
  double *buffer_b = (double *)malloc(panel_K_dim * local_B_cols * sizeof(double));

  /* create derived datatypes to exchange the blocks across the network */
  /* this is due the fact the blocks a process must handle are a portion than the actual dimension of the matrices */
  /* rows of each block are not contiguous in memory */
  MPI_Datatype block_a_type, block_b_type, block_c_type;
  MPI_Type_vector(local_A_rows, panel_K_dim, N, MPI_DOUBLE, &block_a_type);
  MPI_Type_vector(panel_K_dim, local_B_cols, N, MPI_DOUBLE, &block_b_type);
  MPI_Type_vector(local_A_rows, local_B_cols, N, MPI_DOUBLE, &block_c_type);
  MPI_Type_commit(&block_a_type);
  MPI_Type_commit(&block_b_type);
  MPI_Type_commit(&block_c_type);

  for (int k = 0; k < lcm; k++) {
    int sender_column = k % dims[1];
    int sender_row = k % dims[0];

    int block_lda = panel_K_dim;     /* the leading dimension of the block A to use in this step */
    int block_ldb = local_B_cols;    /* the leading dimension of the block B to use in this step */
    const double *block_a, *block_b; /* pointers to the start of the blocks to use in this step */

    if (coords[1] == sender_column) {
      block_a = A;                                                          /* we are sending the block */
      block_lda = N;                                                        /* set the leading dimension to the one of the original matrix */
      A += dims[1] * panel_K_dim;                                           /* we may have to send again in the future, skip the pointer to the start of the other block assigned to the process */
      MPI_Bcast((void *)block_a, 1, block_a_type, sender_column, row_comm); /* send the block as a composite data type, so that multiple lines are received as contiguous */
    } else {
      block_a = buffer_a;                                                                          /* we are receiving, prepare the buffer */
      MPI_Bcast((void *)block_a, local_A_rows * panel_K_dim, MPI_DOUBLE, sender_column, row_comm); /* receive the block as a contiguous array */
    }

    if (coords[0] == sender_row) {
      block_b = B;                                                       /* we are sending the block */
      block_ldb = N;                                                     /* set the leading dimension to the one of the original matrix */
      B += dims[0] * panel_K_dim * N;                                    /* we may have to send again in the future, skip the pointer to the start of the other block assigned to the process */
      MPI_Bcast((void *)block_b, 1, block_b_type, sender_row, col_comm); /* send the block as a composite data type, so that multiple lines are received as contiguous */
    } else {
      block_b = buffer_b;                                                                       /* we are receiving, prepare the buffer */
      MPI_Bcast((void *)block_b, panel_K_dim * local_B_cols, MPI_DOUBLE, sender_row, col_comm); /* receive the block as a contiguous array */
    }

    /* compute product of the blocks */
    f(block_a, block_lda, block_b, block_ldb, offset_c, N, local_A_rows, panel_K_dim, local_B_cols, gpu_count, grid_width, grid_height, block_width);
  }

  if (rank == 0) {
    /* process 0 receives from all other processes */
    for (int i = 1; i < size; i++) {
      int sender_coords[2];
      MPI_Cart_coords(grid_comm, i, 2, sender_coords);

      double *c_dest = C + N * sender_coords[0] * local_A_rows + sender_coords[1] * local_B_cols;
      MPI_Recv(c_dest, 1, block_c_type, i, 0, grid_comm, MPI_STATUS_IGNORE);
    }
  } else {
    /* all other processes send their results to process 0 */
    double *c_start = C + N * coords[0] * local_A_rows + coords[1] * local_B_cols;
    MPI_Send(c_start, 1, block_c_type, 0, 0, grid_comm);
  }

  MPI_Type_free(&block_c_type);
  MPI_Type_free(&block_b_type);
  MPI_Type_free(&block_a_type);

  free(buffer_b);
  free(buffer_a);
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);

  return 0;
}

void phpc_gemm_summa_cuda(MPI_Comm grid_comm, const double *A, const double *B, double *C, int N, int gpu_count, int grid_width, int grid_height, int block_width) {
  phpc_gemm_summa(phpc_gemm_cuda, grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, block_width);
}

void phpc_gemm_summa_cublas(MPI_Comm grid_comm, const double *A, const double *B, double *C, int N, int gpu_count) {
  phpc_gemm_summa(phpc_gemm_cublas, grid_comm, A, B, C, N, gpu_count, 0, 0, 0);
}
