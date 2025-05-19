#include <mpi.h>
#include <stdio.h>

#include "functions.h"
#include "utils.h"

int main(int argc, char *argv[]) {

  int size, rank;
  double *h_A_full = NULL, *h_B_full = NULL, *h_C_full = NULL; 
  int M_A_full, K_A_full, K_B_full, N_B_full;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == 0) {
    printf("Process 0: reading matix A from file...\n");
    initialize_matrix_from_file("inputs/A.bin", &h_A_full, &M_A_full, &K_A_full, rank);

    printf("\nProcess 0: reading matix B from file...\n");
    initialize_matrix_from_file("inputs/B.bin", &h_B_full, &K_B_full, &N_B_full, rank);

    if (K_A_full != K_B_full) {
      fprintf(stderr, "Rank %d: Error: Matrix A's columns (%d) must match Matrix B's rows (%d).\n", rank, K_A_full, K_B_full);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    printf("\nProcess 0: allocatiing and initializing matrix C to 0...\n");
    initialize_matrix_to_zero(&h_C_full, M_A_full, N_B_full, rank);
  }

  MPI_Finalize();
  return 0;
}
