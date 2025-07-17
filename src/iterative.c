#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

void phpc_gemm_iterative(const double *A, const double *B, double *C, int n) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k)
        C[i * n + k] += A[i * n + j] * B[j * n + k];
}

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <matrix_size>>\n", argv[0]);
    return 1;
  }

  int n = atoi(argv[1]);
  double *A = (double *)malloc(n * n * sizeof(double));
  double *B = (double *)malloc(n * n * sizeof(double));
  double *C = (double *)malloc(n * n * sizeof(double));

  assert(A != NULL);
  assert(B != NULL);
  assert(C != NULL);

  for (int i = 0; i < n * n; i++)
    A[i] = B[i] = i;

  memset(C, 0, n * n * sizeof(double));

  double start_time = get_cur_time();
  phpc_gemm_iterative(A, B, C, n);
  double end_time = get_cur_time() - start_time;

  printf("%d,%lf\n", n, end_time);

  free(C);
  free(B);
  free(A);

  return 0;
}
