#include "phpc_matrix_operations.cuh"
#include "utils.cuh"

int main(int argc, char *argv[]) {
  int i, j, Nglob, Mglob, Pglob, ld;
  double *A, *B, *C;
  int dims[2], period[2], coord[2], rank, size;
  dim3 dim_block, dim_grid;
  double start_time, end_time;
  int shared_mem_size;
  int tile_width;
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

  Nglob = 2;
  Mglob = 4;
  Pglob = 4;

  int lcm = find_lcm(dims[0], dims[1]);

  // Possibili dimensioni per i test (il numero di processi utilizzati sarà: 1 4 16 64 256)
  ld = 256;
  // ld = 4096;
  // ld = 8192;
  // ld = 16384;
  // ld = 32768;

  A = (double *)malloc(ld * ld * sizeof(double));
  B = (double *)malloc(ld * ld * sizeof(double));
  C = (double *)malloc(ld * ld * sizeof(double));

  MPI_Cart_coords(grid_comm, rank, 2, coord);

  // ==================================================
  // Test di correttezza
  // ==================================================
  if ((dims[0] == 1 && dims[1] == 1) || (dims[0] == 2 && dims[1] == 2)) {
    Nglob = 2;
    Mglob = 4;
    Pglob = 4;

    for (i = 0; i < Nglob / dims[0]; i++) {
      for (j = 0; j < Mglob / lcm; j++) {
        A[i * ld + j] = 2;
      }
    }

    for (i = 0; i < Mglob / lcm; i++) {
      for (j = 0; j < Pglob / dims[1]; j++) {
        B[i * ld + j] = 2;
      }
    }

    for (i = 0; i < Nglob / dims[0]; i++) {
      for (j = 0; j < Pglob / dims[1]; j++) {
        C[i * ld + j] = 0.0;
      }
    }

    prop = set_gpu_and_get_properties(rank);

    tile_width = 1;
    check_threads_per_block(prop, tile_width, rank);
    check_shared_memory_usage(prop, tile_width, rank);
    dim_block = dim3(tile_width, tile_width, 1);
    dim_grid = dim3(1, 1, 1);
    shared_mem_size = 2 * tile_width * tile_width * sizeof(double);

    phpc_gemm_summa_cuda(grid_comm, A, B, C, ld, Nglob, Mglob, Pglob, dim_block, dim_grid, shared_mem_size);

    MPI_Barrier(grid_comm);

    int test_correctness = 1;
    for (i = 0; i < Nglob / dims[0]; i++) {
      for (j = 0; j < Pglob / dims[1]; j++) {
        if (C[i * ld + j] != 16.0) {
          fprintf(stderr, "Correcteness error at rank %d, C[%d][%d] = %f\n", rank, i, j, C[i * ld + j]);
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
  } else {
    if (rank == 0) printf("Skipping correctness test for grid size %d x %d.\n", dims[0], dims[1]);
  }

  // ==================================================
  // Test di efficienza issata la dimensione del problema (ld x ld) al crescere del numero di thread
  // ==================================================
  prop = set_gpu_and_get_properties(rank);

  srand(0);
  for (i = 0; i < ld; i++) {
    for (j = 0; j < ld; j++) {
      *(A + i * ld + j) = (float)rand() / RAND_MAX;
      *(B + i * ld + j) = (float)rand() / RAND_MAX;
      *(C + i * ld + j) = (float)rand() / RAND_MAX;
    }
  }

  Nglob = ld;  // usiamo tutta la memoria allocata
  double Ndouble = (double)Nglob;
  int local_rows = Nglob / dims[0];
  int local_cols = Nglob / dims[1];

  // Array per memorizzare i risultati dei test
  double test_times[5];
  int test_configs[5][2];  // [tile_width, threads_per_block]
  dim3 test_grids[5];
  double baseline_time = 0.0;

  // ==================================================
  // TEST 1
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
  phpc_gemm_summa_cuda(grid_comm, A, B, C, ld, Nglob, Nglob, Nglob, dim_block, dim_grid, shared_mem_size);
  end_time = get_cur_time() - start_time;

  test_times[0] = end_time;
  test_configs[0][0] = tile_width;
  test_configs[0][1] = dim_block.x * dim_block.y;
  test_grids[0] = dim_grid;
  baseline_time = end_time;

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
  phpc_gemm_summa_cuda(grid_comm, A, B, C, ld, Nglob, Nglob, Nglob, dim_block, dim_grid, shared_mem_size);
  end_time = get_cur_time() - start_time;

  test_times[1] = end_time;
  test_configs[1][0] = tile_width;
  test_configs[1][1] = dim_block.x * dim_block.y;
  test_grids[1] = dim_grid;

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
  phpc_gemm_summa_cuda(grid_comm, A, B, C, ld, Nglob, Nglob, Nglob, dim_block, dim_grid, shared_mem_size);
  end_time = get_cur_time() - start_time;

  test_times[2] = end_time;
  test_configs[2][0] = tile_width;
  test_configs[2][1] = dim_block.x * dim_block.y;
  test_grids[2] = dim_grid;

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
  phpc_gemm_summa_cuda(grid_comm, A, B, C, ld, Nglob, Nglob, Nglob, dim_block, dim_grid, shared_mem_size);
  end_time = get_cur_time() - start_time;

  test_times[3] = end_time;
  test_configs[3][0] = tile_width;
  test_configs[3][1] = dim_block.x * dim_block.y;
  test_grids[3] = dim_grid;

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
  phpc_gemm_summa_cuda(grid_comm, A, B, C, ld, Nglob, Nglob, Nglob, dim_block, dim_grid, shared_mem_size);
  end_time = get_cur_time() - start_time;

  test_times[4] = end_time;
  test_configs[4][0] = tile_width;
  test_configs[4][1] = dim_block.x * dim_block.y;
  test_grids[4] = dim_grid;

  // FILE CSV
  if (rank == 0) {
    FILE *csv_file;
    char filename[256];
    snprintf(filename, sizeof(filename), "csv/performance_%dprocs.csv", size);

    csv_file = fopen(filename, "w");
    if (csv_file == NULL) {
      fprintf(stderr, "Error: Cannot create CSV file %s\n", filename);
    } else {
      // Header del CSV
      fprintf(csv_file, "Test,Processes,Matrix_Size,Tile_Width,Threads_Per_Block,Total_Blocks,Grid_X,Grid_Y,Time_Seconds,GFLOPS,Speedup,Efficiency\n");

      // Calcola GFLOPS di riferimento
      double total_ops = 2.0 * Ndouble * Ndouble * Ndouble;

      for (int test_id = 0; test_id < 5; test_id++) {
        int total_blocks = test_grids[test_id].x * test_grids[test_id].y;
        int total_threads = total_blocks * test_configs[test_id][1];
        double gflops = total_ops / (test_times[test_id] * 1.0e9);
        double speedup = baseline_time / test_times[test_id];
        double efficiency = speedup / total_threads;

        fprintf(csv_file, "%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.2f,%.2f,%.4f\n",
                test_id + 1,               // Test number
                size,                      // Number of MPI processes
                Nglob,                     // Matrix size
                test_configs[test_id][0],  // Tile width
                test_configs[test_id][1],  // Threads per block
                total_blocks,              // Total blocks
                test_grids[test_id].x,     // Grid X dimension
                test_grids[test_id].y,     // Grid Y dimension
                test_times[test_id],       // Time in seconds
                gflops,                    // GFLOPS
                speedup,                   // Speedup vs baseline
                efficiency);               // Efficiency
      }
      fclose(csv_file);
    }
  }

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