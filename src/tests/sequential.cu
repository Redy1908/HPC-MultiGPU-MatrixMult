#include <assert.h>
#include <math.h>
#include <mpi/mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "../phpc_matrix_operations.cuh"

int main(int argc, char **argv) {
  int gpu_count = 1;
  int rank, size;
  int dims[2], period[2], coords[2];
  MPI_Comm grid_comm;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  gpu_count = atoi(argv[1]);
  assert(gpu_count > 0);

  double s = sqrt(size);

  if (s != round(s)) {
    if (rank == 0)
      fprintf(stderr, "Error: Number of processes (%d) must be a perfect square for a square process grid.\n", size);

    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  dims[0] = (int)round(s);
  dims[1] = (int)round(s);

  period[0] = 1;
  period[1] = 1;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 0, &grid_comm);
  MPI_Cart_coords(grid_comm, rank, 2, coords);

  int ld = 1024;
  double *A = (double *)malloc(ld * ld * sizeof(double));
  double *B = (double *)malloc(ld * ld * sizeof(double));
  double *C = (double *)malloc(ld * ld * sizeof(double));

  int width = 8;
  int m = width / dims[0];
  int n = width / dims[1];

  for (size_t i = coords[0] * m; i < coords[0] * m + m; i++) {
    for (size_t j = coords[1] * n; j < coords[1] * n + n; j++) {
      A[i * ld + j] = 2;
      B[i * ld + j] = 2;
    }
  }

  for (size_t i = 0; i < ld * ld; i++)
    C[i] = 0;

  phpc_gemm_summa_cuda(grid_comm, A, B, C, ld, ld, ld, width, gpu_count, 1, 1, 32);

  for (size_t s = 0; s < size; s++) {
    if (rank == s) {
      for (size_t i = 0; i < width; i++) {
        for (size_t j = 0; j < width; j++)
          printf("%.lf ", C[i * ld + j]);

        printf("\n");
      }

      printf("\n");
      fflush(stdout);
    }

    MPI_Barrier(grid_comm);
  }

  free(A);
  free(B);
  free(C);
  MPI_Finalize();

  return 0;
  return 0;
}
