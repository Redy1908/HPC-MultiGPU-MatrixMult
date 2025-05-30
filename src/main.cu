#include "phpc_matrix_operations.cuh"
#include "utils.cuh"

int main(int argc, char *argv[]) {
  int i, j, Nglob, Mglob, Pglob, ld;
  double *A, *B, *C;
  int dims[2], period[2], coord[2], rank, size;
  double time1, time2, Ndouble;
  dim3 dim_block, dim_grid;
  int shared_mem_size;
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

  ld = 6444;
  A = (double *)malloc(ld * ld * sizeof(double));
  B = (double *)malloc(ld * ld * sizeof(double));
  C = (double *)malloc(ld * ld * sizeof(double));

  MPI_Cart_coords(grid_comm, rank, 2, coord);

  // ==================================================
  // Test di correttezza
  // ==================================================
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

  cudaDeviceProp prop = set_gpu_and_get_properties(rank);

  int tile_width = 32;
  check_threads_per_block(prop, tile_width, rank);
  check_shared_memory_usage(prop, tile_width, rank);
  dim_block = dim3(tile_width, tile_width, 1);
  dim_grid = dim3(4, 4, 1);
  shared_mem_size = 2 * tile_width * tile_width * sizeof(double);

  phpc_gemm_summa_cuda(grid_comm, A, B, C, ld, Nglob, Mglob, Pglob, dim_block, dim_grid, shared_mem_size);

  MPI_Barrier(grid_comm);

  for (i = 0; i < Nglob / dims[0]; i++) {
    for (j = 0; j < Pglob / dims[1]; j++) {
      if (C[i * ld + j] != 16.0) {
        fprintf(stderr, "Correcteness error at rank %d, C[%d][%d] = %f\n", rank, i, j, C[i * ld + j]);
      }
    }
  }

  MPI_Barrier(grid_comm);
  if (rank == 0) printf("Corectness test passed.\n");

  // ==================================================
  // Test di efficienza
  // ==================================================

  srand(0);
  for (i = 0; i < ld; i++) {
    for (j = 0; j < ld; j++) {
      *(A + i * ld + j) = (float)rand() / RAND_MAX;
      *(B + i * ld + j) = (float)rand() / RAND_MAX;
      *(C + i * ld + j) = (float)rand() / RAND_MAX;
    }
  }

  if (rank == 0) {
    printf("               N       T   Time       Gflops\n");
  }

  // test di efficienza al crescere delle dimensioni della metrice ed il numero di thread
  for (Nglob = 2048; Nglob <= 2048 * 3; Nglob = Nglob + 2048) {
    Ndouble = Nglob;

    // test con 1 thread
    MPI_Barrier(MPI_COMM_WORLD);
    tile_width = 1;
    check_threads_per_block(prop, tile_width, rank);
    check_shared_memory_usage(prop, tile_width, rank);
    dim_block = dim3(tile_width, tile_width, 1);
    dim_grid = dim3(1, 1, 1);
    shared_mem_size = 2 * tile_width * tile_width * sizeof(double);

    time1 = get_cur_time();
    phpc_gemm_summa_cuda(grid_comm, A, B, C, ld, Nglob, Mglob, Pglob, dim_block, dim_grid, shared_mem_size);
    time2 = get_cur_time() - time1;
    printf(" proc = %d:   %4d   %4d   %e  %f \n", rank, Nglob, dim_block.x * dim_block.y * dim_grid.x * dim_grid.y, time2, 2 * Ndouble * Ndouble * Ndouble / time2 / 1.e9);

    // test con 1024 thread
    MPI_Barrier(MPI_COMM_WORLD);
    tile_width = 32;
    check_threads_per_block(prop, tile_width, rank);
    check_shared_memory_usage(prop, tile_width, rank);
    dim_block = dim3(tile_width, tile_width, 1);
    dim_grid = dim3(1, 1, 1);
    shared_mem_size = 2 * tile_width * tile_width * sizeof(double);

    time1 = get_cur_time();
    phpc_gemm_summa_cuda(grid_comm, A, B, C, ld, Nglob, Mglob, Pglob, dim_block, dim_grid, shared_mem_size);
    time2 = get_cur_time() - time1;
    printf(" proc = %d:   %4d   %4d   %e  %f \n", rank, Nglob, dim_block.x * dim_block.y * dim_grid.x * dim_grid.y, time2, 2 * Ndouble * Ndouble * Ndouble / time2 / 1.e9);

    // Altri test qui chiamare MPI_Barrier(MPI_COMM_WORLD); prima di ogni test

    // questa è la dimensione ottimale della griglia definita sui blocchi locali all'interno di summa dovremmo fare almeno 1 test con la dimensione ottimale da capire come
    // dim3 dim_grid(ceil(block_cols_B / (float)tile_width), ceil(block_rows_A / (float)tile_width), 1);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}