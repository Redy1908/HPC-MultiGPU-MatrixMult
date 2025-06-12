#include <mpi.h>
#include <stdio.h>

#include "phpc_matrix_operations.cuh"
#include "utils.cuh"

int main(int argc, char *argv[]) {
  int i, j, N, local_N;
  double *A, *B, *C;
  int dims[2], period[2], coord[2], rank, size;
  double start_time, end_time;
  int gpu_count, tile_width, grid_width, grid_height;
  char *test_name;
  MPI_Comm grid_comm;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (argc != 6) {
    if (rank == 0) {
      fprintf(stderr, "Usage: %s <matrix_size> <tile_width> <grid_width> <grid_height> <test_name>\n", argv[0]);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  N = atoi(argv[1]);
  tile_width = atoi(argv[2]);
  grid_width = atoi(argv[3]);
  grid_height = atoi(argv[4]);
  test_name = argv[5];

  double s = sqrt(size);

  if (s != round(s)) {
    if (rank == 0) {
      fprintf(stderr, "Error: Number of processes (%d) must be a perfect square for a square process grid.\n", size);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  dims[0] = (int)round(s);
  dims[1] = (int)round(s);

  if (N % dims[0] != 0) {
    if (rank == 0) {
      fprintf(stderr, "Error: Matrix size N (%d) must be divisible by the process grid width (%d).\n", N, dims[0]);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  cudaGetDeviceCount(&gpu_count);

  if (gpu_count <= 0) {
    if (rank == 0) {
      fprintf(stderr, "Error: No CUDA-capable devices found.\n");
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  local_N = N / dims[0];

  if (local_N % gpu_count != 0) {
    if (rank == 0) {
      fprintf(stderr, "Error: Local matrix dimension per rank (%d) must be divisible by gpu_count (%d).\n", local_N, gpu_count);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  period[0] = 1;
  period[1] = 1;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 0, &grid_comm);

  A = (double *)malloc(N * N * sizeof(double));
  B = (double *)malloc(N * N * sizeof(double));
  C = (double *)malloc(N * N * sizeof(double));

  MPI_Cart_coords(grid_comm, rank, 2, coord);

  // ==================================================
  // Test di correttezza
  // ==================================================
  if (rank == 0) printf("\nInitializing matrix A and B with all elements set to 2.0\n");
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      *(A + i * N + j) = 2.0;
      *(B + i * N + j) = 2.0;
      *(C + i * N + j) = 0.0;
    }
  }

  if (rank == 0) printf("\nRunning correctness test...\n");

  if (phpc_gemm_summa_cuda(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width) != 0) {
    fprintf(stderr, "Error in phpc_gemm_summa_cuda\n");
  }

  MPI_Barrier(grid_comm);

  if (rank == 0) {
    printf("Checking correctness...\n");
    int test_correctness = 1;
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        if (C[i * N + j] != 4.0 * N) {
          fprintf(stderr, "Correcteness error at rank %d, C[%d][%d] = %f\n", rank, i, j, C[i * N + j]);
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

    fprintf(csv_file, "matrix_size,n_proc,n_gpu,n_block,n_thread,method,time\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) printf("\nRunning tests:\n");

  // ==================================================
  // TEST Iterative
  // ==================================================
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
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) printf("  Running SUMMA CUDA test...\n");
  memset(C, 0, N * N * sizeof(double));

  start_time = get_cur_time();
  if (phpc_gemm_summa_cuda(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width) != 0) {
    fprintf(stderr, "Error in phpc_gemm_summa_cuda with tile_width = %d\n", tile_width);
  }
  end_time = get_cur_time() - start_time;

  log_to_csv(csv_file, N, size, gpu_count, grid_width * grid_height, tile_width * tile_width, "SUMMA_CUDA", end_time);

  // ==================================================
  // Test SUMMA CUBLAS with the current tile width and grid size
  // ==================================================
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) printf("  Running SUMMA CUBLAS test...\n");
  memset(C, 0, N * N * sizeof(double));

  start_time = get_cur_time();
  if (phpc_gemm_summa_cublas(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width) != 0) {
    fprintf(stderr, "Error in phpc_gemm_summa_cublas with tile_width = %d\n", tile_width);
  }
  end_time = get_cur_time() - start_time;

  log_to_csv(csv_file, N, size, gpu_count, grid_width * grid_height, tile_width * tile_width, "SUMMA_CUBLAS", end_time);

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    fclose(csv_file);
    printf("\n\n  All tests completed.\n\n");
  }

  free(A);
  free(B);
  free(C);

  MPI_Finalize();

  return 0;
}