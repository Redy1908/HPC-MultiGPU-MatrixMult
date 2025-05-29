#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../phpc_matrix_operations.cuh"
#include "../utils.cuh"

int main(int argc, char **argv) {
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int dims[2], kernel_grid_dims[2], block_width;
  int M, K, N;
  MPI_Assert(get_parameters(argc, argv, &M, &K, &N, dims, kernel_grid_dims, &block_width) == 0);
  MPI_Assert(dims[0] * dims[1] == size);

  MPI_Comm grid_comm;
  int period[] = {1, 1}, coord[2];
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 0, &grid_comm);
  MPI_Cart_coords(grid_comm, rank, 2, coord);

  double *A = (double *)malloc(M * K * sizeof(double));
  double *B = (double *)malloc(K * N * sizeof(double));
  double *C = (double *)malloc(M * N * sizeof(double));
  MPI_Assert(A != NULL && B != NULL && C != NULL);

  for (size_t i = 0; i < M * K; i++)
    A[i] = 2;

  for (size_t i = 0; i < N * K; i++)
    B[i] = 2;

  memset(C, 0, sizeof(double) * M * N);

  cudaDeviceProp prop = set_gpu_and_get_properties(rank);
  check_threads_per_block(prop, block_width, rank);   /* FIXME: let this function return an error code and then check */
  check_shared_memory_usage(prop, block_width, rank); /* FIXME: let this function return an error code and then check */

  dim2 grid_size(kernel_grid_dims[0], kernel_grid_dims[1]);
  double start = get_cur_time();

  MPI_Assert(phpc_gemm_summa_cuda(grid_comm, A, B, C, M, K, N, grid_size, block_width) == 0);

  double end = get_cur_time();
  double elapsed = end - start;

  if (rank == 0)
    printf("%lf\n", elapsed);

  free(C);
  free(B);
  free(A);

  MPI_Finalize();

  return 0;
}
