#include <cuda_runtime.h>
#include <stdio.h>

#include "phpc_gemm.cuh"

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

void phpc_gemm_cuda(const double *a, int lda, const double *b, int ldb, double *c, int ldc, int m, int k, int n, int gpu_count, int grid_width, int grid_height, int block_width) {
  int max_shared_memory_per_block;
  cudaDeviceGetAttribute(&max_shared_memory_per_block, cudaDevAttrMaxSharedMemoryPerBlock, 0);

  int required_shared_memory = 2 * block_width * block_width * sizeof(double);

  // if (required_shared_memory > max_shared_memory_per_block)
  //   printf("Warning: required shared memory exceeds the GPU block limit. This will impact performance.\n");

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

  /* unless memory is page-locked, memcpyAsync will default to blocking copy */
  cudaHostRegister((void *)a, m * lda * sizeof(double), cudaHostRegisterPortable);
  cudaHostRegister((void *)b, k * ldb * sizeof(double), cudaHostRegisterPortable);
  cudaHostRegister((void *)c, m * ldc * sizeof(double), cudaHostRegisterPortable);

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

  cudaHostUnregister((void *)c);
  cudaHostUnregister((void *)b);
  cudaHostUnregister((void *)a);

  free(streams);
  free(dev_buffers_c);
  free(dev_buffers_b);
  free(dev_buffers_a);
}
