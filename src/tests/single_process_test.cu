#include <assert.h>
#include <stdio.h>

#include "../phpc_matrix_operations.cuh"
// #include "../utils.cuh"

#define M 512
#define K 1229
#define N 987

int main(int argc, char **argv) {
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

#ifdef CUBLAS
  assert(phpc_gemm_cublas(A, B, C, M, K, N) == 0);
#else
  dim2 grid_size(1, 1);
  assert(phpc_gemm_cuda(A, B, C, M, K, N, grid_size, 32) == 0);
#endif

  for (size_t i = 0; i < N * M; i++)
    printf("%.1lf ", C[i]);

  printf("\n");

  return 0;
}
