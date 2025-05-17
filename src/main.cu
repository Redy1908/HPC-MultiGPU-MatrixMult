#include <mpi.h>
#include <stdio.h>

int get_number_of_gpus() {
  int device_count;
  cudaGetDeviceCount(&device_count);

  return device_count;
}

int main(int argc, char *argv[]) {

  int n_proc, my_id;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

  printf("Process %d of %d i have %d GPUs\n", my_id, n_proc, get_number_of_gpus());

  MPI_Finalize();
}
