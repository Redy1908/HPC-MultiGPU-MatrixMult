#include <cmath>

#include "functions.cuh"
#include "utils.h"

int main(int argc, char *argv[]) {
  int i, j;
  int global_M, global_A_K, global_B_K, global_N, global_K;
  double *h_full_A, *h_full_B, *h_full_C;
  double *h_local_A, *h_local_B, *h_local_C;
  double *all_C_blocks;
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

  MPI_Bcast(&global_M, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_K, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_N, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int local_M = global_M / dims[0];
  // we assume that the process grid is squared so lcm(dims[0], dims[1]) = dims[0] = dims[1]
  int local_K = global_K / dims[1];  // should be global_K / lcm(dims[0], dims[1])
  int local_N = global_N / dims[1];

  int block_size_elements = local_M * local_N;

  if (rank == 0) {
    all_C_blocks = (double *)malloc(size * block_size_elements * sizeof(double));
    MALLOC_CHECK(all_C_blocks, rank, "all_C_blocks");
  }

  h_local_A = (double *)malloc(local_M * local_K * sizeof(double));
  MALLOC_CHECK(h_local_A, rank, "h_local_A");

  h_local_B = (double *)malloc(local_K * local_N * sizeof(double));
  MALLOC_CHECK(h_local_B, rank, "h_local_B");

  h_local_C = (double *)malloc(local_M * local_N * sizeof(double));
  MALLOC_CHECK(h_local_C, rank, "h_local_C");

  // Scatter matric A
  if (rank == 0) {
    double *send_buffer_A = (double *)malloc((size_t)local_M * local_K * sizeof(double));
    MALLOC_CHECK(send_buffer_A, rank, "send_buffer_A for A distribution");

    for (int target_p_rank = 0; target_p_rank < size; ++target_p_rank) {
      int current_process_coords[2];
      MPI_Cart_coords(grid_comm, target_p_rank, 2, current_process_coords);

      int p_coord_row = current_process_coords[0];
      int p_coord_col = current_process_coords[1];

      int start_row_global_A = p_coord_row * local_M;
      int start_col_global_A = p_coord_col * local_K;

      for (i = 0; i < local_M; ++i) {
        for (j = 0; j < local_K; ++j) {
          send_buffer_A[i * local_K + j] = h_full_A[(start_row_global_A + i) * global_K + (start_col_global_A + j)];
        }
      }

      if (target_p_rank == 0) {
        for (i = 0; i < local_M * local_K; ++i) {
          h_local_A[i] = send_buffer_A[i];
        }
      } else {
        MPI_Send(send_buffer_A, local_M * local_K, MPI_DOUBLE, target_p_rank, 0, MPI_COMM_WORLD);
      }
    }
    free(send_buffer_A);
  } else {
    MPI_Recv(h_local_A, local_M * local_K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Scatter matrix B
  if (rank == 0) {
    double *send_buffer_B = (double *)malloc((size_t)local_K * local_N * sizeof(double));
    MALLOC_CHECK(send_buffer_B, rank, "send_buffer_B for B distribution");

    for (int target_p_rank = 0; target_p_rank < size; ++target_p_rank) {
      int current_process_coords[2];
      MPI_Cart_coords(grid_comm, target_p_rank, 2, current_process_coords);

      int p_coord_row = current_process_coords[0];
      int p_coord_col = current_process_coords[1];

      int start_row_global_B = p_coord_row * local_K;
      int start_col_global_B = p_coord_col * local_N;

      for (i = 0; i < local_K; ++i) {
        for (j = 0; j < local_N; ++j) {
          send_buffer_B[i * local_N + j] = h_full_B[(start_row_global_B + i) * global_N + (start_col_global_B + j)];
        }
      }

      if (target_p_rank == 0) {
        for (i = 0; i < local_K * local_N; ++i) {
          h_local_B[i] = send_buffer_B[i];
        }
      } else {
        MPI_Send(send_buffer_B, local_K * local_N, MPI_DOUBLE, target_p_rank, 1, MPI_COMM_WORLD);
      }
    }
    free(send_buffer_B);
  } else {
    MPI_Recv(h_local_B, local_K * local_N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  cudaDeviceProp prop = set_gpu_and_get_properties(rank);

  int tile_width = 32;
  check_threads_per_block(prop, tile_width, rank);
  check_shared_memory_usage(prop, tile_width, rank);

  SUMMA(grid_comm, h_local_A, h_local_B, h_local_C, global_M, global_K, global_N, tile_width, rank);

  MPI_Gather(h_local_C, block_size_elements, MPI_DOUBLE,
             all_C_blocks, block_size_elements, MPI_DOUBLE,
             0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int p_rank = 0; p_rank < size; p_rank++) {
      int p_coord_row = p_rank / dims[1];
      int p_coord_col = p_rank % dims[1];

      int start_row_global = p_coord_row * local_M;
      int start_col_global = p_coord_col * local_N;

      double *source_block_ptr = all_C_blocks + p_rank * block_size_elements;

      for (int i = 0; i < local_M; i++) {
        for (int j = 0; j < local_N; j++) {
          int global_row_idx = start_row_global + i;
          int global_col_idx = start_col_global + j;

          if (global_row_idx < global_M && global_col_idx < global_N) {
            h_full_C[global_row_idx * global_N + global_col_idx] = source_block_ptr[i * local_N + j];
          }
        }
      }
    }

    printf("\nResult\n");
    for (i = 0; i < global_M; i++) {
      for (j = 0; j < global_N; j++) {
        printf("%lf ", h_full_C[i * global_N + j]);
      }
      printf("\n");
    }
  }

  MPI_Finalize();
  return 0;
}
