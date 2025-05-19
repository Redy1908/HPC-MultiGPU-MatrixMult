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
  double *A, *B, *C;
  int M_A, K_A, K_B, N_B;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == 0) {
    printf("Process 0: reading matix A from file...\n");
    initialize_matrix_from_file("inputs/A.bin", &A, &M_A, &K_A);

    printf("\nProcess 0: reading matix B from file...\n");
    initialize_matrix_from_file("inputs/B.bin", &B, &K_B, &N_B);

    printf("\nProcess 0: allocatiing and initializing matrix C to 0...\n");
    initialize_matrix_to_zero(&C, M_A, N_B);

  }

  MPI_Finalize();
}
