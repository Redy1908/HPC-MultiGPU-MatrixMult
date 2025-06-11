#include <assert.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "../phpc_matrix_operations.cuh"

#define MPI_ASSERT(condition)                  \
  {                                            \
    if (!(condition)) {                        \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
    }                                          \
  }

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm grid_comm;
  int dims[2], period[2], rank;

  int width = atoi(argv[1]);
  dims[0] = atoi(argv[2]);
  dims[1] = atoi(argv[3]);
  period[0] = period[1] = 1;

  int gpus;
  cudaGetDeviceCount(&gpus);
  MPI_ASSERT(gpus > 0);
  MPI_ASSERT(width >= gpus);

  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 0, &grid_comm);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double *a = (double *)malloc(width * width * sizeof(double));
  double *b = (double *)malloc(width * width * sizeof(double));
  double *c = (double *)malloc(width * width * sizeof(double));

  MPI_ASSERT(a != NULL && b != NULL && c != NULL);

  for (size_t i = 0; i < width * width; i++) {
    a[i] = rank + 1;
    b[i] = rank + 1;
  }

  int status = phpc_gemm_summa_cublas(grid_comm, a, b, c, width, gpus, 1, 1, 32);
  MPI_ASSERT(status == 0);

  if (rank == 0) {
    for (size_t i = 0; i < width; i++) {
      for (size_t j = 0; j < width; j++)
        printf("%.1lf ", c[i * width + j]);

      printf("\n");
    }
  }

  free(c);
  free(b);
  free(a);

  MPI_Finalize();

  return 0;
}
