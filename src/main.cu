#include <mpi.h>
#include <stdio.h>

#include "functions.h"
#include "utils.h"

#define MALLOC_CHECK(ptr, rank_main, var_name_str)                                     \
  if (ptr == NULL) {                                                                   \
    fprintf(stderr, "MALLOC Error in %s at line %d (Rank %d): Failed to allocate %s\n",\
            __FILE__, __LINE__, rank_main, var_name_str);                              \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                           \
  }

#define CUDA_CHECK(err, rank)                                                         \
  if (err != cudaSuccess) {                                                           \
    fprintf(stderr, "CUDA Error in %s at line %d (Rank %d): %s\n", __FILE__, __LINE__,\
            rank, cudaGetErrorString(err));                                           \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                          \
  }

int get_number_of_gpus() {
  int device_count;
  cudaGetDeviceCount(&device_count);

  return device_count;
}

int main(int argc, char *argv[]) {

  int size, rank;
  double *h_A_full = NULL, *h_B_full = NULL, *h_C_full = NULL; 
  int M_A, K_A, K_B, N_B;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == 0) {
    printf("Process 0: reading matix A from file...\n");
    initialize_matrix_from_file("inputs/A.bin", &h_A_full, &M_A, &K_A, rank);

    printf("\nProcess 0: reading matix B from file...\n");
    initialize_matrix_from_file("inputs/B.bin", &h_B_full, &K_B, &N_B, rank);

    if (K_A != K_B) {
      fprintf(stderr, "Rank %d: Error: Matrix A's columns (%d) must match Matrix B's rows (%d).\n", rank, K_A, K_B);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    printf("\nProcess 0: allocatiing and initializing matrix C to 0...\n");
    initialize_matrix_to_zero(&h_C_full, M_A, N_B, rank);
  }

  MPI_Finalize();
  return 0;
}
