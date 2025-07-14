#include <cuda_runtime.h>
#include <math.h>
#include <mpi.h>
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
  int i, j;
  int dims[2], period[2], coord[2], rank, size;
  double start_time, end_time;

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

  // ==================================================
  // Test di correttezza
  // ==================================================
  // FIXME: don't need to run this test every time once we estabilished it works
  if (rank == 0) {
    printf("\nInitializing matrix A and B to 2.0, matrix C to 0.0...\n");
  }
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      *(A + i * N + j) = 2.0;
      *(B + i * N + j) = 2.0;
      *(C + i * N + j) = 0.0;
    }
  }
  if (rank == 0) {
    printf("Matrix initialization complete.\n");
    printf("\nRunning correctness test...\n");
  }

  phpc_gemm_summa_cuda(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width);

  if (rank == 0) {
    printf("Checking correctness...\n");
    int test_correctness = 1;
    double expected_value = 4.0 * N;
    double epsilon = 1e-9;

    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        if (fabs(C[i * N + j] - expected_value) > epsilon) {
          fprintf(stderr, "Correcteness error at rank %d, C[%d][%d] = %f, expected %f\n", rank, i, j, C[i * N + j], expected_value);
          test_correctness = 0;
        }
      }
    }

    if (test_correctness) {
      printf("Correctness test passed.\n");
    } else {
      printf("Correctness test FAILED. Aborting...\n");
      free(A);
      free(B);
      free(C);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  // ==================================================
  // Test di efficienza
  // ==================================================

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

  if (rank == 0) {
    printf("\nInitializing matrix A and B to random values, matrix C to 0.0...\n");
  }
  srand(0);
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      *(A + i * N + j) = (double)rand() / RAND_MAX;
      *(B + i * N + j) = (double)rand() / RAND_MAX;
    }
  }

  if (rank == 0) {
    printf("Matrix initialization complete.\n");
    printf("\nRunning tests:\n");
  }

  // ==================================================
  // TEST Iterative
  // ==================================================
  // FIXME: don't need to run this test every time if the only thing that changes is the matrices size
  if (rank == 0) {
    printf("  Running iterative test...\n");
    memset(C, 0, N * N * sizeof(double));

    start_time = get_cur_time();
    phpc_gemm_iterative(A, B, C, N);
    end_time = get_cur_time() - start_time;

    log_to_csv(csv_file, N, 1, gpu_count, 0, 0, "ITERATIVE", end_time);
  }

  // ==================================================
  // Test SUMMA CUDA
  // ==================================================
  if (rank == 0) printf("  Running SUMMA CUDA test...\n");
  memset(C, 0, N * N * sizeof(double));
  MPI_Barrier(MPI_COMM_WORLD);

  start_time = get_cur_time();
  phpc_gemm_summa_cuda(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width);
  end_time = get_cur_time() - start_time;

  if (rank == 0)
    log_to_csv(csv_file, N, size, gpu_count, grid_width * grid_height, tile_width * tile_width, "SUMMA_CUDA", end_time);

  // ==================================================
  // Test SUMMA CUBLAS with the current tile width and grid size
  // ==================================================
  if (rank == 0) printf("  Running SUMMA CUBLAS test...\n");
  memset(C, 0, N * N * sizeof(double));
  MPI_Barrier(MPI_COMM_WORLD);

  start_time = get_cur_time();
  phpc_gemm_summa_cublas(grid_comm, A, B, C, N, gpu_count);
  end_time = get_cur_time() - start_time;

  if (rank == 0) {
    log_to_csv(csv_file, N, size, gpu_count, grid_width * grid_height, tile_width * tile_width, "SUMMA_CUBLAS", end_time);
    fclose(csv_file);
    printf("\n  All tests completed.\n\n");
  }

  free(A);
  free(B);
  free(C);

  MPI_Finalize();

  return 0;
}