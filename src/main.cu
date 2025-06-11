#include <mpi.h>
#include <stdio.h>

#include "phpc_matrix_operations.cuh"
#include "utils.cuh"

int main(int argc, char *argv[]) {
  int i, j, N, local_N, local_N_gpu;
  double *A, *B, *C;
  int dims[2], period[2], coord[2], rank, size;
  double start_time, end_time;
  int tile_width, grid_width, grid_height;
  int gpu_count;
  MPI_Comm grid_comm;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (argc < 3) {
    if (rank == 0) {
      fprintf(stderr, "Usage: %s <matrix_size> <num_tile_width> <tile_width1> <tile_width2> ...\n", argv[0]);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  N = atoi(argv[1]);
  int num_tile_widths = atoi(argv[2]);

  if (argc < 3 + num_tile_widths) {
    if (rank == 0) {
      fprintf(stderr, "Error: Not enough tile_width arguments provided. Expected %d, got %d.\n", num_tile_widths, argc - 3);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  int *tile_widths_array = (int *)malloc(num_tile_widths * sizeof(int));

  for (int k = 0; k < num_tile_widths; ++k) {
    tile_widths_array[k] = atoi(argv[3 + k]);
    if (tile_widths_array[k] <= 0) {
      if (rank == 0) {
        fprintf(stderr, "Error: Invalid tile_width value %s at index %d.\n", argv[3 + k], k);
      }
      free(tile_widths_array);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

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
  local_N_gpu = local_N / gpu_count;

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
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      *(A + i * N + j) = 2.0;
      *(B + i * N + j) = 2.0;
      *(C + i * N + j) = 0.0;
    }
  }

  tile_width = 32;
  if (rank == 0) printf("Running correctness test: tile_width = %d...\n", tile_width);

  grid_width = (unsigned int)ceil((double)local_N_gpu / tile_width);
  grid_height = (unsigned int)ceil((double)local_N / tile_width);

  if (phpc_gemm_summa_cuda(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width) != 0) {
    fprintf(stderr, "Error in phpc_gemm_summa_cuda\n");
  }

  MPI_Barrier(grid_comm);

  int test_correctness = 1;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      if (C[i * N + j] != 4.0 * N) {
        fprintf(stderr, "Correcteness error at rank %d, C[%d][%d] = %f\n", rank, i, j, C[i * N + j]);
        test_correctness = 0;
      }
    }
  }

  int global_test_passed = 0;
  MPI_Allreduce(&test_correctness, &global_test_passed, 1, MPI_INT, MPI_MIN, grid_comm);

  if (rank == 0) {
    if (global_test_passed) {
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
  // Test di efficienza fissata la dimensione del problema (N x N) al crescere del numero di thread
  // ==================================================

  srand(0);
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      *(A + i * N + j) = (double)rand() / RAND_MAX;
      *(B + i * N + j) = (double)rand() / RAND_MAX;
      *(C + i * N + j) = 0.0;
    }
  }

  FILE *csv_file = NULL;
  char filename[256];
  if (rank == 0) {
    snprintf(filename, sizeof(filename), "csv/performanceN%d_%dtasks_%dgpus.csv", N, size, gpu_count);
    csv_file = fopen(filename, "w");
    if (csv_file == NULL) {
      fprintf(stderr, "Error: Could not create CSV file %s\n", filename);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    fprintf(csv_file, "N,n_proc,n_block,n_thread,method,time\n");
  }

  // ==================================================
  // TEST Iterative
  // ==================================================
  if (rank == 0) {
    printf("\nRunning Test 0: Iterative on rank 0...\n");

    memset(C, 0, N * N * sizeof(double));

    start_time = get_cur_time();
    phpc_gemm_iterative(A, B, C, N);
    end_time = get_cur_time() - start_time;

    log_to_csv(csv_file, N, 1, 0, 0, "ITERATIVE", end_time);
  }

  // ==================================================
  // TESTS with increasing tile widths
  // ==================================================
  for (i = 0; i < num_tile_widths; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    tile_width = tile_widths_array[i];

    if (rank == 0) printf("\nRunning Test %d: tile_width = %d...\n", i + 1, tile_width);

    // If tile_width is 1 we are testing with only one thread (1x1 grid with one 1x1 block) otherwise we are
    // using the optimal grid size based on tile_width.
    grid_width = (tile_width == 1) ? 1 : (unsigned int)ceil((double)local_N_gpu / tile_width);
    grid_height = (tile_width == 1) ? 1 : (unsigned int)ceil((double)local_N / tile_width);

    // ==================================================
    // Test SUMMA CUDA with the current tile width and grid size
    // ==================================================
    start_time = get_cur_time();
    if (phpc_gemm_summa_cuda(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width) != 0) {
      fprintf(stderr, "Error in phpc_gemm_summa_cuda with tile_width = %d\n", tile_width);
    }
    end_time = get_cur_time() - start_time;

    log_to_csv(csv_file, N, size, grid_width * grid_height, tile_width * tile_width, "SUMMA_CUDA", end_time);

    // ==================================================
    // Test SUMMA CUBLAS with the current tile width and grid size
    // ==================================================
    MPI_Barrier(MPI_COMM_WORLD);
    memset(C, 0, N * N * sizeof(double));

    start_time = get_cur_time();
    if (phpc_gemm_summa_cublas(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width) != 0) {
      fprintf(stderr, "Error in phpc_gemm_summa_cublas with tile_width = %d\n", tile_width);
    }
    end_time = get_cur_time() - start_time;

    log_to_csv(csv_file, N, size, grid_width * grid_height, tile_width * tile_width, "SUMMA_CUBLAS", end_time);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    fclose(csv_file);
    printf("\nAll tests completed.\n");
  }

  free(A);
  free(B);
  free(C);

  MPI_Finalize();

  return 0;
}