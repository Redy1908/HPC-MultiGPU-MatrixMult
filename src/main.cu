#include <stdio.h>

#include "phpc_matrix_operations.cuh"
#include "utils.cuh"

int main(int argc, char *argv[]) {
  int i, j, N;
  double *A, *B, *C;
  int dims[2], period[2], coord[2], rank, size;
  dim3 dim_block, dim_grid;
  double start_time, end_time;
  int shared_mem_size;
  int tile_width;
  int gpu_count;
  cudaDeviceProp prop;
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

  // Possibili dimensioni per i test (il numero di processi utilizzati sarà: 1 4 16)
  N = 256;
  // N = 4096;
  // N = 8192;
  // N = 16384;
  // N = 32768;

  A = (double *)malloc(N * N * sizeof(double));
  B = (double *)malloc(N * N * sizeof(double));
  C = (double *)malloc(N * N * sizeof(double));

  int local_rows = N / dims[0];
  int local_cols = N / dims[1];

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

  gpu_count = get_number_of_gpus();
  prop = set_gpu_and_get_properties(rank);

  tile_width = 32;
  check_threads_per_block(prop, tile_width, rank);
  check_shared_memory_usage(prop, tile_width, rank);
  dim_block = dim3(tile_width, tile_width, 1);

  dim_grid.x = (unsigned int)ceil((double)local_cols / dim_block.x);
  dim_grid.y = (unsigned int)ceil((double)local_rows / dim_block.y);
  dim_grid.z = 1;

  shared_mem_size = 2 * tile_width * tile_width * sizeof(double);

  phpc_gemm_summa_cuda(grid_comm, A, B, C, N, N, N, N, gpu_count, dim_grid.x, dim_grid.y, tile_width);

  MPI_Barrier(grid_comm);

  int test_correctness = 1;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      if (C[i * N + j] != 4 * N) {
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
      printf("Correctness test FAILED.\n");
    }
  }

  // ==================================================
  // Test di efficienza fissata la dimensione del problema (N x N) al crescere del numero di thread
  // ==================================================

  srand(0);
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      *(A + i * N + j) = (float)rand() / RAND_MAX;
      *(B + i * N + j) = (float)rand() / RAND_MAX;
      *(C + i * N + j) = 0.0;
    }
  }

  /* Per salvare i risultati dei test in un file CSV al variare del numero di processi inizializzare il file nel seguente modo:
   *
   * I file csv saranno in csv/
   *
   * FILE *csv_file;
   * char filename[256];
   * snprintf(filename, sizeof(filename), "csv/performance_%dprocs_%dgpu.csv", size, gpu_count);
   * csv_file = fopen(filename, "w");
   */

  // ==================================================
  // TEST 1 - 1 solo thread
  // ==================================================
  if (rank == 0) printf("Running Test 1: tile_width = 1...\n");
  MPI_Barrier(MPI_COMM_WORLD);
  tile_width = 1;
  check_threads_per_block(prop, tile_width, rank);
  check_shared_memory_usage(prop, tile_width, rank);
  dim_block = dim3(tile_width, tile_width, 1);
  dim_grid = dim3(1, 1, 1);
  shared_mem_size = 2 * tile_width * tile_width * sizeof(double);

  start_time = get_cur_time();
  phpc_gemm_summa_cuda(grid_comm, A, B, C, N, N, N, N, 1, dim_grid.x, dim_grid.y, tile_width);
  end_time = get_cur_time() - start_time;

  // ==================================================
  // TEST 2
  // ==================================================
  if (rank == 0) printf("Running Test 2: tile_width = 4...\n");
  MPI_Barrier(MPI_COMM_WORLD);
  tile_width = 4;
  check_threads_per_block(prop, tile_width, rank);
  check_shared_memory_usage(prop, tile_width, rank);
  dim_block = dim3(tile_width, tile_width, 1);

  dim_grid.x = (unsigned int)ceil((double)local_cols / dim_block.x);
  dim_grid.y = (unsigned int)ceil((double)local_rows / dim_block.y);
  dim_grid.z = 1;

  shared_mem_size = 2 * tile_width * tile_width * sizeof(double);

  start_time = get_cur_time();
  // phpc_gemm_summa_cuda(grid_comm, A, B, C, N, dim_block, dim_grid, shared_mem_size);
  phpc_gemm_summa_cuda(grid_comm, A, B, C, N, N, N, N, gpu_count, dim_grid.x, dim_grid.y, tile_width);
  end_time = get_cur_time() - start_time;

  // ==================================================
  // TEST 3
  // ==================================================
  if (rank == 0) printf("Running Test 3: tile_width = 8...\n");
  MPI_Barrier(MPI_COMM_WORLD);
  tile_width = 8;
  check_threads_per_block(prop, tile_width, rank);
  check_shared_memory_usage(prop, tile_width, rank);
  dim_block = dim3(tile_width, tile_width, 1);

  dim_grid.x = (unsigned int)ceil((double)local_cols / dim_block.x);
  dim_grid.y = (unsigned int)ceil((double)local_rows / dim_block.y);
  dim_grid.z = 1;

  shared_mem_size = 2 * tile_width * tile_width * sizeof(double);

  start_time = get_cur_time();
  phpc_gemm_summa_cuda(grid_comm, A, B, C, N, N, N, N, gpu_count, dim_grid.x, dim_grid.y, tile_width);
  end_time = get_cur_time() - start_time;

  // ==================================================
  // TEST 4
  // ==================================================
  if (rank == 0) printf("Running Test 4: tile_width = 16...\n");
  MPI_Barrier(MPI_COMM_WORLD);
  tile_width = 16;
  check_threads_per_block(prop, tile_width, rank);
  check_shared_memory_usage(prop, tile_width, rank);
  dim_block = dim3(tile_width, tile_width, 1);

  dim_grid.x = (unsigned int)ceil((double)local_cols / dim_block.x);
  dim_grid.y = (unsigned int)ceil((double)local_rows / dim_block.y);
  dim_grid.z = 1;

  shared_mem_size = 2 * tile_width * tile_width * sizeof(double);

  start_time = get_cur_time();
  phpc_gemm_summa_cuda(grid_comm, A, B, C, N, N, N, N, gpu_count, dim_grid.x, dim_grid.y, tile_width);
  end_time = get_cur_time() - start_time;

  // ==================================================
  // TEST 5
  // ==================================================
  if (rank == 0) printf("Running Test 5: tile_width = 32..\n");
  MPI_Barrier(MPI_COMM_WORLD);
  tile_width = 32;
  check_threads_per_block(prop, tile_width, rank);
  check_shared_memory_usage(prop, tile_width, rank);
  dim_block = dim3(tile_width, tile_width, 1);

  dim_grid.x = (unsigned int)ceil((double)local_cols / dim_block.x);
  dim_grid.y = (unsigned int)ceil((double)local_rows / dim_block.y);
  dim_grid.z = 1;

  shared_mem_size = 2 * tile_width * tile_width * sizeof(double);

  start_time = get_cur_time();
  phpc_gemm_summa_cuda(grid_comm, A, B, C, N, N, N, N, gpu_count, dim_grid.x, dim_grid.y, tile_width);
  end_time = get_cur_time() - start_time;

  MPI_Barrier(MPI_COMM_WORLD);

  free(A);
  free(B);
  free(C);

  if (rank == 0) {
    printf("All tests completed.\n");
  }

  MPI_Finalize();

  return 0;
}