#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void gemm_kernel(float *A, float *B, float *C, int M, int N, int K) {
  extern __shared__ float shared_mem[];

  int tile_width = blockDim.x;

  float *s_A = shared_mem;
  float *s_B = shared_mem + tile_width * tile_width;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int num_tiles_along_N = (int)ceil((float)N / tile_width);
  int num_tiles_along_M = (int)ceil((float)M / tile_width);
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

    float c_value = 0;

    int num_phases = (int)ceil((float)K / tile_width);
    for (int phase = 0; phase < num_phases; ++phase) {
      if ((global_row_C < M) && (phase * tile_width + tx) < K)
        s_A[ty * tile_width + tx] = A[global_row_C * K + phase * tile_width + tx];
      else
        s_A[ty * tile_width + tx] = 0;

      if ((phase * tile_width + ty) < K && (global_col_C < N))
        s_B[ty * tile_width + tx] = B[(phase * tile_width + ty) * N + global_col_C];
      else
        s_B[ty * tile_width + tx] = 0;

      __syncthreads();

      for (int k_tile = 0; k_tile < tile_width; ++k_tile)
        c_value += s_A[ty * tile_width + k_tile] * s_B[k_tile * tile_width + tx];

      __syncthreads();
    }

    if (global_row_C < M && global_col_C < N)
      C[global_row_C * N + global_col_C] += c_value;
  }
}

int phpc_gemm_cuda(const float *a, const float *b, float *c, int width, int gpu_count, int grid_width, int grid_height, int block_width) {
  int device_count;
  cudaGetDeviceCount(&device_count);

  assert(width > 0);
  assert(gpu_count > 0);
  assert(grid_width > 0 && grid_height > 0);
  assert(block_width * block_width <= 1024);
  assert(gpu_count <= device_count);
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
  int n = width / gpu_count;
  int k = width;

  float **dev_buffers_a = (float **)malloc(gpu_count * sizeof(float *));
  float **dev_buffers_b = (float **)malloc(gpu_count * sizeof(float *));
  float **dev_buffers_c = (float **)malloc(gpu_count * sizeof(float *));
  cudaStream_t *streams = (cudaStream_t *)malloc(gpu_count * sizeof(cudaStream_t));

  assert(dev_buffers_a != NULL);
  assert(dev_buffers_b != NULL);
  assert(dev_buffers_c != NULL);
  assert(streams != NULL);

  for (int i = 0; i < gpu_count; i++) {
    cudaSetDevice(i);
    cudaMalloc(&(dev_buffers_a[i]), m * k * sizeof(float));
    cudaMalloc(&(dev_buffers_b[i]), k * n * sizeof(float));
    cudaMalloc(&(dev_buffers_c[i]), k * n * sizeof(float));
    cudaStreamCreate(&(streams[i]));

    /* copy column of B*/
    cudaMemcpy2DAsync(dev_buffers_b[i], n * sizeof(float), b + n * i, width * sizeof(float), n * sizeof(float), width, cudaMemcpyHostToDevice, streams[i]);

    /* initialize device C to 0 */
    cudaMemset(dev_buffers_c[i], 0, n * k * sizeof(float));
  }

  dim3 grid_size(grid_width, grid_height, 1);
  dim3 block_size(block_width, block_width, 1);
  int shared_memory_size = 2 * block_width * block_width * sizeof(float);

  for (int i = 0; i < gpu_count; i++) {
    for (size_t j = 0; j < gpu_count; j++) {
      cudaSetDevice(j);

      /* copy row j % gpu_count of A */
      int row = (j + i) % gpu_count;
      cudaMemcpyAsync(dev_buffers_a[j], a + m * k * row, m * k * sizeof(float), cudaMemcpyHostToDevice, streams[j]);

      int dev_offset_c = m * n * row;
      gemm_kernel<<<grid_size, block_size, shared_memory_size, streams[j]>>>(dev_buffers_a[j], dev_buffers_b[j], dev_buffers_c[j] + dev_offset_c, m, n, k);
    }
  }

  /**
   * Gather results: each GPU has a column of the resulting matrix
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
    cudaMemcpy2D(c + n * i, width * sizeof(float), dev_buffers_c[i], n * sizeof(float), n * sizeof(float), width, cudaMemcpyDeviceToHost);

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

int main(int argc, char const *argv[]) {
  int width = atoi(argv[1]);
  int gpus = atoi(argv[2]);

  assert(gpus > 0);
  assert(width >= gpus);

  float *a = (float *)malloc(width * width * sizeof(float));
  float *b = (float *)malloc(width * width * sizeof(float));
  float *c = (float *)malloc(width * width * sizeof(float));

  assert(a != NULL && b != NULL && c != NULL);

  for (size_t i = 0; i < width * width; i++) {
    a[i] = i + 1;
    b[i] = i + 1;
  }

  phpc_gemm_cuda(a, b, c, width, gpus, 1, 1, 32);

  for (size_t i = 0; i < width; i++) {
    for (size_t j = 0; j < width; j++)
      printf("%.1f ", c[i * width + j]);

    printf("\n");
  }

  free(c);
  free(b);
  free(a);

  return 0;
}
