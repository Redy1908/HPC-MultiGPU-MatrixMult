#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Comm grid_comm;
  int dims[2] = {0, 0}, periods[2] = {0, 0};

  MPI_Dims_create(size, 2, dims);
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

  if (rank == 0)
    printf("Grid: %d %d\n", dims[0], dims[1]);

  MPI_Finalize();

  return 0;
}
