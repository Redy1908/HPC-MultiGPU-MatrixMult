#include <assert.h>
#include <stdio.h>

#include "../phpc_matrix_operations.cuh"
#include "../utils.cuh"

int main(int argc, char **argv) {
  int dims[2], kernel_grid_dims[2], block_width;
  int M, K, N;
  get_parameters(argc, argv, &M, &K, &N, dims, kernel_grid_dims, &block_width);

  double *A = (double *)malloc(M * K * sizeof(double));
  double *B = (double *)malloc(K * N * sizeof(double));
  double *C = (double *)malloc(N * M * sizeof(double));
  assert(A != NULL && B != NULL && C != NULL);

  for (size_t i = 0; i < M * K; i++) {
    A[i] = i + 1;
  }

  for (size_t i = 0; i < N * K; i++) {
    B[i] = i + 1;
  }

  memset(C, 0, sizeof(double) * M * N);

  double start = get_cur_time();

#ifdef CUBLAS
  printf("cuBLAS ");
  assert(phpc_gemm_cublas(A, B, C, M, K, N) == 0);
#else
#ifdef CUDA
  printf("custom-kernel ");
  dim2 grid_size(kernel_grid_dims[0], kernel_grid_dims[1]);
  assert(phpc_gemm_cuda(A, B, C, M, K, N, grid_size, block_width) == 0);
#else
  printf("sequential ");
  assert(phpc_gemm_sequential(A, B, C, M, K, N) == 0);
#endif
#endif

  double end = get_cur_time();
  double elapsed = end - start;
  printf("%lf\n", elapsed);

  return 0;
}
