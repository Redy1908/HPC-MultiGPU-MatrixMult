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
  int threads_per_block, num_blocks;
  int gpu_count;
  MPI_Comm grid_comm;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (argc < 2) {
    if (rank == 0) {
      fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  N = atoi(argv[1]);
  if (N <= 0) {
    if (rank == 0) {
      fprintf(stderr, "Error: Invalid matrix size N = %d. Must be > 0.\n", N);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
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
  // TEST 1 - 1 solo thread
  // ==================================================
  MPI_Barrier(MPI_COMM_WORLD);
  tile_width = 1;
  if (rank == 0) printf("\nRunning Test 1: tile_width = %d...\n", tile_width);

  grid_width = (unsigned int)ceil((double)local_N_gpu / tile_width);
  grid_height = (unsigned int)ceil((double)local_N / tile_width);

  memset(C, 0, N * N * sizeof(double));

  start_time = get_cur_time();
  if (phpc_gemm_summa_cuda(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width) != 0) {
    fprintf(stderr, "Error in phpc_gemm_summa_cuda with tile_width = %d\n", tile_width);
  }
  end_time = get_cur_time() - start_time;

  if (rank == 0 && csv_file != NULL) {
    threads_per_block = tile_width * tile_width;
    num_blocks = grid_width * grid_height;
    fprintf(csv_file, "%d,%d,%d,%d,SUMMA_CUDA,%f\n",
            N, size, num_blocks, threads_per_block, end_time);
  }

  memset(C, 0, N * N * sizeof(double));

  start_time = get_cur_time();
  if (phpc_gemm_summa_cublas(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width) != 0) {
    fprintf(stderr, "Error in phpc_gemm_summa_cublas with tile_width = %d\n", tile_width);
  }
  end_time = get_cur_time() - start_time;

  if (rank == 0 && csv_file != NULL) {
    threads_per_block = tile_width * tile_width;
    num_blocks = grid_width * grid_height;
    fprintf(csv_file, "%d,%d,%d,%d,SUMMA_CUBLAS,%f\n",
            N, size, num_blocks, threads_per_block, end_time);
  }

  // ==================================================
  // TEST 2
  // ==================================================
  MPI_Barrier(MPI_COMM_WORLD);
  tile_width = 4;
  if (rank == 0) printf("\nRunning Test 2: tile_width = %d...\n", tile_width);

  grid_width = (unsigned int)ceil((double)local_N_gpu / tile_width);
  grid_height = (unsigned int)ceil((double)local_N / tile_width);

  memset(C, 0, N * N * sizeof(double));

  start_time = get_cur_time();
  if (phpc_gemm_summa_cuda(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width) != 0) {
    fprintf(stderr, "Error in phpc_gemm_summa_cuda with tile_width = %d\n", tile_width);
  }
  end_time = get_cur_time() - start_time;

  if (rank == 0 && csv_file != NULL) {
    threads_per_block = tile_width * tile_width;
    num_blocks = grid_width * grid_height;
    fprintf(csv_file, "%d,%d,%d,%d,SUMMA_CUDA,%f\n",
            N, size, num_blocks, threads_per_block, end_time);
  }

  memset(C, 0, N * N * sizeof(double));

  start_time = get_cur_time();
  if (phpc_gemm_summa_cublas(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width) != 0) {
    fprintf(stderr, "Error in phpc_gemm_summa_cublas with tile_width = %d\n", tile_width);
  }
  end_time = get_cur_time() - start_time;

  if (rank == 0 && csv_file != NULL) {
    threads_per_block = tile_width * tile_width;
    num_blocks = grid_width * grid_height;
    fprintf(csv_file, "%d,%d,%d,%d,SUMMA_CUBLAS,%f\n",
            N, size, num_blocks, threads_per_block, end_time);
  }

  // ==================================================
  // TEST 3
  // ==================================================
  MPI_Barrier(MPI_COMM_WORLD);
  tile_width = 8;
  if (rank == 0) printf("\nRunning Test 3: tile_width = %d...\n", tile_width);

  grid_width = (unsigned int)ceil((double)local_N_gpu / tile_width);
  grid_height = (unsigned int)ceil((double)local_N / tile_width);

  memset(C, 0, N * N * sizeof(double));

  start_time = get_cur_time();
  if (phpc_gemm_summa_cuda(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width) != 0) {
    fprintf(stderr, "Error in phpc_gemm_summa_cuda with tile_width = %d\n", tile_width);
  }
  end_time = get_cur_time() - start_time;

  if (rank == 0 && csv_file != NULL) {
    threads_per_block = tile_width * tile_width;
    num_blocks = grid_width * grid_height;
    fprintf(csv_file, "%d,%d,%d,%d,SUMMA_CUDA,%f\n",
            N, size, num_blocks, threads_per_block, end_time);
  }

  memset(C, 0, N * N * sizeof(double));

  start_time = get_cur_time();
  if (phpc_gemm_summa_cublas(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width) != 0) {
    fprintf(stderr, "Error in phpc_gemm_summa_cublas with tile_width = %d\n", tile_width);
  }
  end_time = get_cur_time() - start_time;

  if (rank == 0 && csv_file != NULL) {
    threads_per_block = tile_width * tile_width;
    num_blocks = grid_width * grid_height;
    fprintf(csv_file, "%d,%d,%d,%d,SUMMA_CUBLAS,%f\n",
            N, size, num_blocks, threads_per_block, end_time);
  }

  // ==================================================
  // TEST 4
  // ==================================================
  MPI_Barrier(MPI_COMM_WORLD);

  tile_width = 16;
  if (rank == 0) printf("\nRunning Test 4: tile_width = %d...\n", tile_width);

  grid_width = (unsigned int)ceil((double)local_N_gpu / tile_width);
  grid_height = (unsigned int)ceil((double)local_N / tile_width);

  memset(C, 0, N * N * sizeof(double));

  start_time = get_cur_time();
  if (phpc_gemm_summa_cuda(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width) != 0) {
    fprintf(stderr, "Error in phpc_gemm_summa_cuda with tile_width = %d\n", tile_width);
  }
  end_time = get_cur_time() - start_time;

  if (rank == 0 && csv_file != NULL) {
    threads_per_block = tile_width * tile_width;
    num_blocks = grid_width * grid_height;
    fprintf(csv_file, "%d,%d,%d,%d,SUMMA_CUDA,%f\n",
            N, size, num_blocks, threads_per_block, end_time);
  }

  memset(C, 0, N * N * sizeof(double));

  start_time = get_cur_time();
  if (phpc_gemm_summa_cublas(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width) != 0) {
    fprintf(stderr, "Error in phpc_gemm_summa_cublas with tile_width = %d\n", tile_width);
  }
  end_time = get_cur_time() - start_time;

  if (rank == 0 && csv_file != NULL) {
    threads_per_block = tile_width * tile_width;
    num_blocks = grid_width * grid_height;
    fprintf(csv_file, "%d,%d,%d,%d,SUMMA_CUBLAS,%f\n",
            N, size, num_blocks, threads_per_block, end_time);
  }

  // ==================================================
  // TEST 5
  // ==================================================
  MPI_Barrier(MPI_COMM_WORLD);

  tile_width = 32;
  if (rank == 0) printf("\nRunning Test 5: tile_width = %d...\n", tile_width);

  grid_width = (unsigned int)ceil((double)local_N_gpu / tile_width);
  grid_height = (unsigned int)ceil((double)local_N / tile_width);

  memset(C, 0, N * N * sizeof(double));

  start_time = get_cur_time();
  if (phpc_gemm_summa_cuda(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width) != 0) {
    fprintf(stderr, "Error in phpc_gemm_summa_cuda with tile_width = %d\n", tile_width);
  }
  end_time = get_cur_time() - start_time;

  if (rank == 0 && csv_file != NULL) {
    threads_per_block = tile_width * tile_width;
    num_blocks = grid_width * grid_height;
    fprintf(csv_file, "%d,%d,%d,%d,SUMMA_CUDA,%f\n",
            N, size, num_blocks, threads_per_block, end_time);
  }

  memset(C, 0, N * N * sizeof(double));

  start_time = get_cur_time();
  if (phpc_gemm_summa_cublas(grid_comm, A, B, C, N, gpu_count, grid_width, grid_height, tile_width) != 0) {
    fprintf(stderr, "Error in phpc_gemm_summa_cublas with tile_width = %d\n", tile_width);
  }
  end_time = get_cur_time() - start_time;

  if (rank == 0 && csv_file != NULL) {
    threads_per_block = tile_width * tile_width;
    num_blocks = grid_width * grid_height;
    fprintf(csv_file, "%d,%d,%d,%d,SUMMA_CUBLAS,%f\n",
            N, size, num_blocks, threads_per_block, end_time);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0 && csv_file != NULL) {
    fclose(csv_file);
    printf("\nCSV file written: %s\n", filename);
  }

  if (rank == 0) {
    printf("\nAll tests completed.\n");
  }

  free(A);
  free(B);
  free(C);

  MPI_Finalize();

  return 0;
}