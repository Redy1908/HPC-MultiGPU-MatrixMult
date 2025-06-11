#include <assert.h>
#include <mpi/mpi.h>

#include "../phpc_matrix_operations.cuh"

int main(int argc, char **argv) {
  assert(argc >= 5);

  MPI_Init(&argc, &argv);

  MPI_Comm grid_comm;
  int dims[2], period[2], rank;

  int width = atoi(argv[1]);
  int gpus = atoi(argv[2]);
  dims[0] = atoi(argv[3]);
  dims[1] = atoi(argv[4]);
  period[0] = period[1] = 1;

  assert(gpus > 0);
  assert(width >= gpus);

  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 0, &grid_comm);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  float *a = (float *)malloc(width * width * sizeof(float));
  float *b = (float *)malloc(width * width * sizeof(float));
  float *c = (float *)malloc(width * width * sizeof(float));

  assert(a != NULL && b != NULL && c != NULL);

  for (size_t i = 0; i < width * width; i++) {
    a[i] = rank + 1;
    b[i] = rank + 1;
  }

  int status = phpc_gemm_summa_cublas(grid_comm, a, b, c, width, gpus, 1, 1, 1024);
  assert(status == 0);

  if (rank == 0) {
    for (size_t i = 0; i < width; i++) {
      for (size_t j = 0; j < width; j++)
        printf("%.1f ", c[i * width + j]);

      printf("\n");
    }
  }

  free(c);
  free(b);
  free(a);

  MPI_Finalize();

  return 0;
}
