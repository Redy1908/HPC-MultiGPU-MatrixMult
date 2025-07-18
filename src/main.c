#include <cuda_runtime.h>
#include <math.h>
#include <mpi/mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "phpc_summa.h"
#include "utils.h"

#define MPI_ASSERT(check)                                              \
  if (!(check)) {                                                      \
    fprintf(stderr, "Check at " __FILE__ " line %d failed", __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                           \
  }

int main(int argc, char *argv[]) {
  int dims[2], period[2], coord[2], rank, size;
  double start_time, cuda_time, cublas_time;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (argc != 6) {
    if (rank == 0) {
      fprintf(stderr, "Usage: %s <matrix_size> <tile_width> <grid_width> <grid_height> <test_name>\n", argv[0]);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  int N = atoi(argv[1]);
  int tile_width = atoi(argv[2]);
  int grid_width = atoi(argv[3]);
  int grid_height = atoi(argv[4]);
  char *test_name = argv[5];

  if (size == 1) {
    dims[0] = 1;
    dims[1] = 1;
  } else {
    dims[0] = 0;
    dims[1] = 0;
    MPI_Dims_create(size, 2, dims);
  }

  if (N % dims[0] != 0 || N % dims[1] != 0) {
    if (rank == 0) {
      fprintf(stderr, "Error: Matrix size N (%d) must be divisible by process grid dimensions (%d x %d).\n", N, dims[0], dims[1]);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  int gpu_count;
  cudaGetDeviceCount(&gpu_count);
  MPI_ASSERT(gpu_count > 0);

  period[0] = 1;
  period[1] = 1;
  MPI_Comm grid_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 0, &grid_comm);
  MPI_Cart_coords(grid_comm, rank, 2, coord);

  double *A = (double *)malloc(N * N * sizeof(double));
  double *B = (double *)malloc(N * N * sizeof(double));
  double *C = (double *)malloc(N * N * sizeof(double));

  MPI_ASSERT(A != NULL);
  MPI_ASSERT(B != NULL);
  MPI_ASSERT(C != NULL);

  FILE *csv_file = NULL;
  char filename[256];
  if (rank == 0) {
    snprintf(filename, sizeof(filename), "csv/%s_N%d_T%d_G%d_TW%d_GW%d_GH%d.csv", test_name, N, size, gpu_count, tile_width, grid_width, grid_height);
    csv_file = fopen(filename, "w");
    if (csv_file == NULL) {
      fprintf(stderr, "Error: Could not create CSV file %s\n", filename);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // fprintf(csv_file, "matrix_size,n_proc,n_gpu,n_block,n_thread_per_block,method,time\n");
  }

  for (int i = 0; i < N * N; i++)
    A[i] = B[i] = i;

  /* Test kernel */

  float cuda_gpu_time;
  MPI_Barrier(MPI_COMM_WORLD);

  start_time = get_cur_time();
  phpc_gemm_summa_cuda(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width, &cuda_gpu_time);
  cuda_time = get_cur_time() - start_time;

  MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &cuda_gpu_time, &cuda_gpu_time, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  cuda_gpu_time /= size;

  /* Test cuBLAS */

  float cublas_gpu_time;
  MPI_Barrier(MPI_COMM_WORLD);

  start_time = get_cur_time();
  phpc_gemm_summa_cublas(grid_comm, A, B, C, N, gpu_count, &cublas_gpu_time);
  cublas_time = get_cur_time() - start_time;

  /* Cleanup */

  if (rank == 0) {
    log_to_csv(csv_file, N, size, gpu_count, grid_width * grid_height, tile_width * tile_width, cuda_time, cuda_gpu_time, cublas_time);
    fclose(csv_file);
  }

  free(A);
  free(B);
  free(C);

  MPI_Finalize();

  return 0;
}