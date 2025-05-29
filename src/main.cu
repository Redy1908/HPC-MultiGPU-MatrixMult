#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "phpc_matrix_operations.cuh"
#include "utils.cuh"

int main(int argc, char *argv[]) {
  int i, j, M, K, N;
  double *A, *B, *C;
  int dims[2], period[2], coord[2], rank, size;
  MPI_Comm grid_comm;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double s = sqrt(size);

  if (s != round(s)) {
    if (rank == 0) {
      fprintf(stderr, "Error: Number of processes (%d) must be a perfect square for a square process grid.\n", size);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  dims[0] = (int)round(s);
  dims[1] = (int)round(s);

  period[0] = 1;
  period[1] = 1;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 0, &grid_comm);
  MPI_Cart_coords(grid_comm, rank, 2, coord);

  read_matrix_dimensions("inputs/A.bin", &M, &K, rank);
  read_matrix_dimensions("inputs/B.bin", &K, &N, rank);

  int lcm = find_lcm(dims[0], dims[1]);

  C = (double *)malloc(M * N * sizeof(double));
  MALLOC_CHECK(C, rank, "C");

  int local_M = M / dims[0];
  int local_K = K / lcm;
  int local_N = N / dims[1];

  read_matrix_A_block("inputs/A.bin", &A, M, K, local_M, local_K, coord[0], lcm, rank);
  read_matrix_B_block("inputs/B.bin", &B, K, N, local_K, local_N, coord[1], lcm, rank);
  memset(C, 0, sizeof(double) * M * N);

  cudaDeviceProp prop = set_gpu_and_get_properties(rank);

  dim2 grid_size(1, 1);
  unsigned int block_width = 4;

  check_threads_per_block(prop, block_width, rank);
  check_shared_memory_usage(prop, block_width, rank);

  double start = get_cur_time();

  phpc_gemm_summa_cuda(grid_comm, A, B, C, M, K, N, grid_size, block_width);

  double end = get_cur_time();
  double elapsed = end - start;

  if (rank == 0) {
    printf("Result\n");
    for (i = 0; i < M; i++) {
      for (j = 0; j < N; j++)
        printf("%lf ", C[i * N + j]);

      printf("\n");
    }
  }

  free(A);
  free(B);
  free(C);

  MPI_Finalize();
  return 0;
}