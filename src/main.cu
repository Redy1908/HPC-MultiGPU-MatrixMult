#include "phpc_matrix_operations.cuh"
#include "utils.cuh"

int main(int argc, char *argv[]) {
  int i, j, Nglob, Mglob, Pglob, ld;
  double *A, *B, *C;
  int dims[2], period[2], coord[2], rank, size;
  double start_time, end_time;
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

  int lcm = find_lcm(dims[0], dims[1]);

  ld = 6444;
  A = (double *)malloc(ld * ld * sizeof(double));
  B = (double *)malloc(ld * ld * sizeof(double));
  C = (double *)malloc(ld * ld * sizeof(double));

  MPI_Cart_coords(grid_comm, rank, 2, coord);

  // ==================================================
  // Test di correttezza
  // ==================================================
  Nglob = 32;
  Mglob = 32;
  Pglob = 32;

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
  dim_grid = dim3(1, 1, 1);
  shared_mem_size = 2 * tile_width * tile_width * sizeof(double);

  phpc_gemm_summa_cuda(grid_comm, A, B, C, ld, ld, ld, Nglob, Mglob, Pglob, dim_block, dim_grid, shared_mem_size);

  MPI_Barrier(grid_comm);

  int test_correctness = 1;
  for (i = 0; i < Nglob / dims[0]; i++) {
    for (j = 0; j < Pglob / dims[1]; j++) {
      if (C[i * ld + j] != 128.0) {
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

  // test di efficienza al crescere delle dimensioni della metrice ed il numero di thread
  for (Nglob = 2048; Nglob <= 2048 * 3; Nglob = Nglob + 2048) {
    /*
     * Test con 1 thread, bisogna considerare che questo test sarà essere molto lento, attualmente lo script limita l'esecuzione a 5 minuti
     * consiglio di rimuove questo test per testare il codice
     */
    MPI_Barrier(MPI_COMM_WORLD);
    tile_width = 1;
    check_threads_per_block(prop, tile_width, rank);
    check_shared_memory_usage(prop, tile_width, rank);
    dim_block = dim3(tile_width, tile_width, 1);
    dim_grid = dim3(1, 1, 1);  // forziamo 1 solo blocco per utilizzare 1 solo thread
    shared_mem_size = 2 * tile_width * tile_width * sizeof(double);

    start_time = get_cur_time();
    phpc_gemm_summa_cuda(grid_comm, A, B, C, ld, ld, ld, Nglob, Nglob, Nglob, dim_block, dim_grid, shared_mem_size);
    end_time = get_cur_time() - start_time;

    /*
     * Test con 1024 thread per blocco (GPU LIMIT) e grid size ottimale
     *
     * Questo test dovrebbe darci il risultato migliore considerando una matrice abbastanza grande
     *
     * La dimensione della griglia viene calcolata automaticamente in modo ottimale dentro summa nel
     * seguente modo: dim_grid.x = (unsigned int)ceil((double)local_B_cols / dim_block.x);
     *                grid.y = (unsigned int)ceil((double)local_A_rows / dim_block.y);
     *
     * considerando le dimensioni effettive delle porzioni di matrici che verranno moltiplicate.)
     *
     * Ad esempio: consideriamo 2 blocchi di matrici 2048x2048, con tile_width = 32, avremo una griglia di 64x64 blocchi
     * da 1024 thread ciascuno, quindi 64*64*1024 = 4194304 thread. Questo ci permette di mappare 1:1 i blocchi di matrice
     * con i thread della GPU, e di avere il massimo parallelismo possibile.
     *
     */
    MPI_Barrier(MPI_COMM_WORLD);
    tile_width = 32;                                    // block size 32x32 = 1024 threads
    check_threads_per_block(prop, tile_width, rank);    // controlliamo lo stesso di non superare i 1024 threads
    check_shared_memory_usage(prop, tile_width, rank);  // controlliamo di avere abbastanza shared memory per copiarci 2 tile 32x32 di double
    dim_block = dim3(tile_width, tile_width, 1);
    dim_grid = dim3(0, 0, 0);  // quando passiamo dim3(0,0,0) la dimensione della griglia viene calcolata automaticamente in modo ottimale dentro summa
    shared_mem_size = 2 * tile_width * tile_width * sizeof(double);

    start_time = get_cur_time();
    phpc_gemm_summa_cuda(grid_comm, A, B, C, ld, ld, ld, Nglob, Nglob, Nglob, dim_block, dim_grid, shared_mem_size);
    end_time = get_cur_time() - start_time;

    // Altri test qui chiamare MPI_Barrier(MPI_COMM_WORLD); prima di ogni test
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