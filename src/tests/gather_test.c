#include <assert.h>
#include <mpi/mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define M 7
#define K 20
#define N 10

int main(int argc, char **argv) {
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Comm grid_comm;
  int period[] = {1, 1}, dims[] = {2, 3}, coord[2];
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 0, &grid_comm);
  MPI_Cart_coords(grid_comm, rank, 2, coord);

  int *recv_counts = rank == 0 ? malloc(dims[0] * sizeof(int)) : NULL;
  int *recv_displs = rank == 0 ? malloc(dims[0] * sizeof(int)) : NULL;

  int remain_dims_row[2] = {0, 1};
  int remain_dims_col[2] = {1, 0};
  MPI_Comm row_comm, col_comm;
  MPI_Cart_sub(grid_comm, remain_dims_row, &row_comm);
  MPI_Cart_sub(grid_comm, remain_dims_col, &col_comm);

  int local_M = M / dims[0] + (coord[0] < M % dims[0]);
  int local_N = N / dims[1] + (coord[1] < N % dims[1]);

  int buffer_M = rank == 0 ? M : local_M;
  int buffer_N = coord[1] == 0 ? N : local_N;

  double *C = (double *)malloc(buffer_M * buffer_N * sizeof(double));
  assert(C != NULL);

  for (size_t i = 0; i < buffer_M * buffer_N; i++)
    C[i] = rank + 1;

  for (size_t i = 0; i < size; i++) {
    if (rank == i) {
      for (int i = 0; i < buffer_M; i++) {
        for (int j = 0; j < buffer_N; j++)
          printf("%lf ", C[i * buffer_N + j]);

        printf("\n");
      }
    }

    MPI_Barrier(grid_comm);
  }

  /* send row */

  if (coord[1] == 0) {
    if (rank == 0) {
      recv_displs[0] = 0;
      recv_counts[0] = local_M * N;

      for (size_t i = 1; i < dims[0]; i++) {
        int coord[2];
        MPI_Cart_coords(col_comm, i, 2, coord);

        int local_M = M / dims[0] + (coord[0] < M % dims[0]);

        recv_counts[i] = local_M * N;
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
      }
    }

    MPI_Gatherv(C, local_M * N, MPI_DOUBLE, C, recv_counts, recv_displs, MPI_DOUBLE, 0, col_comm);
    // if (coord[0] == 0) {
    //   MPI_Gather(C, , MPI_DOUBLE, C + local_M * N, (M - local_M) * N, MPI_DOUBLE, 0, col_comm);
    // } else {
    //   MPI_Gather(NULL, 0, MPI_DOUBLE, C + local_M * N, (M - local_M) * N, MPI_DOUBLE, 0, col_comm);
    // }
  }

  //   if (rank == 0) {
  //     for (int i = 0; i < M; i++) {
  //       for (int j = 0; j < N; j++)
  //         printf("%lf ", C[i * N + j]);

  //       printf("\n");
  //     }
  //   }

  free(C);

  MPI_Finalize();

  return 0;
}
