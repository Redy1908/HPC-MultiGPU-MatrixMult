#include <cublasXt.h>
#include <cuda_runtime.h>

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
    int phases = K / tile_width + (K % tile_width != 0);

    for (int phase = 0; phase < phases; ++phase) {
      __syncthreads();

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
    }

    if (global_row_C < M && global_col_C < N)
      C[global_row_C * N + global_col_C] += c_value;
  }
}

void phpc_gemm_cuda(const double *a, int lda, const double *b, int ldb, double *c, int ldc, int m, int k, int n, int gpu_count, int grid_width, int grid_height, int block_width, float *compute_time) {
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
  int required_shared_memory = 2 * block_width * block_width * sizeof(double);

  double **dev_buffers_a = (double **)malloc(gpu_count * sizeof(double *));
  double **dev_buffers_b = (double **)malloc(gpu_count * sizeof(double *));
  double **dev_buffers_c = (double **)malloc(gpu_count * sizeof(double *));
  cudaStream_t *streams = (cudaStream_t *)malloc(gpu_count * sizeof(cudaStream_t));
  cudaEvent_t *events = (cudaEvent_t *)malloc(gpu_count * 2 * sizeof(cudaEvent_t));

  /* unless memory is page-locked, memcpyAsync will default to blocking copy */
  cudaHostRegister((void *)a, m * lda * sizeof(double), cudaHostRegisterPortable);
  cudaHostRegister((void *)b, k * ldb * sizeof(double), cudaHostRegisterPortable);
  cudaHostRegister((void *)c, m * ldc * sizeof(double), cudaHostRegisterPortable);

  for (int gpu = 0; gpu < gpu_count; gpu++) {
    int dev_n = n / gpu_count + (gpu < n % gpu_count);

    cudaSetDevice(gpu);
    cudaStreamCreate(&(streams[gpu]));

    cudaEventCreate(events + 2 * gpu);
    cudaEventCreate(events + 2 * gpu + 1);

    cudaMallocAsync(&(dev_buffers_a[gpu]), m * k * sizeof(double), streams[gpu]);
    cudaMallocAsync(&(dev_buffers_b[gpu]), k * dev_n * sizeof(double), streams[gpu]);
    cudaMallocAsync(&(dev_buffers_c[gpu]), m * dev_n * sizeof(double), streams[gpu]);

    /* copy from host to device */
    cudaMemcpy2DAsync(dev_buffers_a[gpu], k * sizeof(double), a, lda * sizeof(double), k * sizeof(double), m, cudaMemcpyHostToDevice, streams[gpu]);
    cudaMemcpy2DAsync(dev_buffers_b[gpu], dev_n * sizeof(double), b, ldb * sizeof(double), dev_n * sizeof(double), k, cudaMemcpyHostToDevice, streams[gpu]);
    cudaMemcpy2DAsync(dev_buffers_c[gpu], dev_n * sizeof(double), c, ldc * sizeof(double), dev_n * sizeof(double), m, cudaMemcpyHostToDevice, streams[gpu]);

    /* perform computation */
    cudaEventRecord(events[2 * gpu], streams[gpu]);
    gemm_kernel<<<grid_size, block_size, required_shared_memory, streams[gpu]>>>(dev_buffers_a[gpu], dev_buffers_b[gpu], dev_buffers_c[gpu], m, dev_n, k);
    cudaEventRecord(events[2 * gpu + 1], streams[gpu]);

    /* copy result from device to host */
    cudaMemcpy2DAsync(c, ldc * sizeof(double), dev_buffers_c[gpu], dev_n * sizeof(double), dev_n * sizeof(double), m, cudaMemcpyDeviceToHost, streams[gpu]);

    cudaFreeAsync(dev_buffers_c[gpu], streams[gpu]);
    cudaFreeAsync(dev_buffers_b[gpu], streams[gpu]);
    cudaFreeAsync(dev_buffers_a[gpu], streams[gpu]);

    b += dev_n;
    c += dev_n;
  }

  float average_elapsed = 0;
  for (int gpu = 0; gpu < gpu_count; gpu++) {
    cudaSetDevice(gpu);
    cudaStreamSynchronize(streams[gpu]);

    float ms = 0;
    cudaEventElapsedTime(&ms, events[2 * gpu], events[2 * gpu + 1]);
    cudaEventDestroy(events[2 * gpu]);
    cudaEventDestroy(events[2 * gpu + 1]);
    average_elapsed += ms;

    cudaStreamDestroy(streams[gpu]);
  }

  *compute_time = average_elapsed / (gpu_count * 1000);

  cudaHostUnregister((void *)c);
  cudaHostUnregister((void *)b);
  cudaHostUnregister((void *)a);

  free(events);
  free(streams);
  free(dev_buffers_c);
  free(dev_buffers_b);
  free(dev_buffers_a);
}

void phpc_gemm_cublas(const double *a, int lda, const double *b, int ldb, double *c, int ldc, int m, int k, int n, int gpu_count, int grid_width, int grid_height, int block_width, float *gpu_time) {
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

  *gpu_time = 0;
}
