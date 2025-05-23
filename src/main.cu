#include "functions.cuh"
#include "utils.h"

int main(int argc, char *argv[]) {
  int i, j, M, K, N, p_rank;
  double *A, *B, *C, *all_C_blocks;
  int dims[2], period[2], coord[2], rank, size;
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
  MPI_Cart_coords(grid_comm, rank, 2, coord);

  /* INIZIALIZZAZIONE MATRICI DA FILE
    if (rank == 0) {
        printf("RANK 0: Reading matrix A from file...\n");
        initialize_matrix_from_file("inputs/A.bin", &h_full_A, &global_M, &global_A_K, rank);
        printf("RANK 0: Matrix A (%dx%d) read successfully. \n", global_M, global_A_K);

        printf("\nRANK 0: Reading matrix B from file...\n");
        initialize_matrix_from_file("inputs/B.bin", &h_full_B, &global_B_K, &global_N, rank);
        printf("RANK 0: Matrix B (%dx%d) read successfully.\n", global_B_K, global_N);

        if (global_A_K != global_B_K) {
        fprintf(stderr, "Error: Matrix A and B dimensions do not match.\n");
        free(h_full_A);
        free(h_full_B);
        MPI_Finalize();
        return -1;
        }

        global_K = global_A_K;

        // we assume that the process grid is squared so lcm(dims[0], dims[1]) = dims[0] = dims[1]
        if (global_M % dims[0] != 0 || global_K % dims[0] != 0 || global_N % dims[1] != 0) {
        fprintf(stderr, "Error: Matrix dimensions are not divisible by the grid dimensions.\n");
        free(h_full_A);
        free(h_full_B);
        MPI_Finalize();
        return -1;
        }

        initialize_matrix_to_zero(&h_full_C, global_M, global_N, rank);
    }
  */

  M = 2;
  K = 4;
  N = 4;

  int lcm = find_lcm(dims[0], dims[1]);

  A = (double *)malloc(M * K * sizeof(double));
  MALLOC_CHECK(A, rank, "A");

  B = (double *)malloc(K * N * sizeof(double));
  MALLOC_CHECK(B, rank, "B");

  C = (double *)malloc(M * N * sizeof(double));
  MALLOC_CHECK(C, rank, "C");

  int local_M = M / dims[0];
  int local_K = K / lcm;
  int local_N = N / dims[1];

  int block_size_elements = local_M * local_N;

  if (rank == 0) {
    all_C_blocks = (double *)malloc(size * block_size_elements * sizeof(double));
    MALLOC_CHECK(all_C_blocks, rank, "all_C_blocks");
  }

  for (i = 0; i < local_M; i++) {
    for (j = 0; j < local_K; j++) {
      A[i * K + j] = coord[0] * dims[1] * local_K + coord[1] * local_K + i * K + j;
    }
  }

  for (i = 0; i < local_K; i++) {
    for (j = 0; j < local_N; j++) {
      B[i * N + j] = 10 + coord[0] * dims[1] * N + coord[1] * local_N + i * N + j;
    }
  }

  for (i = 0; i < local_M; i++) {
    for (j = 0; j < local_N; j++) {
      C[i * N + j] = 0.0;
    }
  }

  cudaDeviceProp prop = set_gpu_and_get_properties(rank);

  int tile_width = 32;
  check_threads_per_block(prop, tile_width, rank);
  check_shared_memory_usage(prop, tile_width, rank);

  SUMMA(grid_comm, A, B, C, M, K, N, tile_width, rank);

  MPI_Gather(C, block_size_elements, MPI_DOUBLE,
             all_C_blocks, block_size_elements, MPI_DOUBLE,
             0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (p_rank = 0; p_rank < size; p_rank++) {
      int p_coord_row = p_rank / dims[1];
      int p_coord_col = p_rank % dims[1];

      int start_row_global = p_coord_row * local_M;
      int start_col_global = p_coord_col * local_N;

      double *source_block_ptr = all_C_blocks + p_rank * block_size_elements;

      for (i = 0; i < local_M; i++) {
        for (j = 0; j < local_N; j++) {
          int global_row_idx = start_row_global + i;
          int global_col_idx = start_col_global + j;

          if (global_row_idx < M && global_col_idx < N) {
            C[global_row_idx * N + global_col_idx] = source_block_ptr[i * local_N + j];
          }
        }
      }
    }

    printf("Result\n");
    for (i = 0; i < M; i++) {
      for (j = 0; j < N; j++) {
        printf("%lf ", C[i * N + j]);
      }
      printf("\n");
    }
  }

  free(A);
  free(B);
  free(C);

  if (rank == 0)
    free(all_C_blocks);

  MPI_Finalize();
  return 0;
}