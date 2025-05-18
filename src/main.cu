#include <mpi.h>
#include <stdio.h>

#include "functions.h"
#include "utils.h"

int get_number_of_gpus() {
  int device_count;
  cudaGetDeviceCount(&device_count);

  return device_count;
}

int main(int argc, char *argv[]) {

  int size, rank;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  printf("Process %d of %d i have %d GPUs\n", rank + 1, size, get_number_of_gpus());

  MPI_Finalize();
}
