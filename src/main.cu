#include <string.h>

#include "phpc_matrix_operations.cuh"
#include "utils.cuh"

int main(int argc, char **argv) {
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int dims[2], kernel_grid_dims[2], block_width;
  int M, K, N;
  MPI_Assert(get_parameters(argc, argv, &M, &K, &N, dims, kernel_grid_dims, &block_width) == 0);
  MPI_Assert(dims[0] * dims[1] == size);

  MPI_Comm grid_comm;
  int period[] = {1, 1}, coord[2];
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 0, &grid_comm);
  MPI_Cart_coords(grid_comm, rank, 2, coord);

  // int M, K, N, p_rank;
  // read_matrix_dimensions("inputs/A.bin", &M, &K, rank);
  // read_matrix_dimensions("inputs/B.bin", &K, &N, rank);

  int lcm = find_lcm(dims[0], dims[1]);

  double *A, *B, *all_C_blocks;
  double *C = (double *)malloc(M * N * sizeof(double));
  MPI_Assert(C != NULL);

  int local_M = M / dims[0];
  int local_K = K / lcm;
  int local_N = N / dims[1];

  int block_size_elements = local_M * local_N;

  if (rank == 0) {
    all_C_blocks = (double *)malloc(size * block_size_elements * sizeof(double));
    MALLOC_CHECK(all_C_blocks, rank, "all_C_blocks");
  }

  // read_matrix_A_block("inputs/A.bin", &A, M, K, local_M, local_K, coord[0], lcm, rank);
  // read_matrix_B_block("inputs/B.bin", &B, K, N, local_K, local_N, coord[1], lcm, rank);
  A = (double *)malloc(M * K * sizeof(double));
  B = (double *)malloc(K * N * sizeof(double));
  MPI_Assert(A != NULL && B != NULL);

  for (size_t i = 0; i < M * K; i++) {
    A[i] = rank + 1;
  }

  for (size_t i = 0; i < N * K; i++) {
    B[i] = rank + 1;
  }

  memset(C, 0, sizeof(double) * M * N);

  cudaDeviceProp prop = set_gpu_and_get_properties(rank);

  check_threads_per_block(prop, block_width, rank);
  check_shared_memory_usage(prop, block_width, rank);

  dim2 grid_size(kernel_grid_dims[0], kernel_grid_dims[1]);
  double start = get_cur_time();

  phpc_gemm_summa_cuda(grid_comm, A, B, C, M, K, N, grid_size, block_width);

  if (rank == 0) {
    double end = get_cur_time();
    double elapsed = end - start;
    printf("%lf\n", elapsed);
  }

  // MPI_Gather(C, block_size_elements, MPI_DOUBLE,
  //            all_C_blocks, block_size_elements, MPI_DOUBLE,
  //            0, MPI_COMM_WORLD);

  // for (size_t i = 0; i < size; i++) {
  //   if (rank == i) {
  //     unsigned int rows = M / dims[0] + (coord[0] < M % dims[0]);
  //     unsigned int columns = N / dims[1] + (coord[1] < N % dims[1]);

  //     printf("Process %d\n", rank);
  //     for (int i = 0; i < rows; i++) {
  //       for (int j = 0; j < columns; j++)
  //         printf("%lf ", C[i * columns + j]);

  //       printf("\n");
  //     }
  //   }

  //   MPI_Barrier(grid_comm);
  // }

  // if (rank == 0) {
  //   for (p_rank = 0; p_rank < size; p_rank++) {
  //     int p_coord_row = p_rank / dims[1];
  //     int p_coord_col = p_rank % dims[1];

  //     int start_row_global = p_coord_row * local_M;
  //     int start_col_global = p_coord_col * local_N;

  //     double *source_block_ptr = all_C_blocks + p_rank * block_size_elements;

  //     int i, j;
  //     for (i = 0; i < local_M; i++) {
  //       for (j = 0; j < local_N; j++) {
  //         int global_row_idx = start_row_global + i;
  //         int global_col_idx = start_col_global + j;

  //         if (global_row_idx < M && global_col_idx < N) {
  //           C[global_row_idx * N + global_col_idx] = source_block_ptr[i * local_N + j];
  //         }
  //       }
  //     }
  //   }

  //   int i, j;
  //   printf("Result\n");
  //   for (i = 0; i < M; i++) {
  //     for (j = 0; j < N; j++)
  //       printf("%lf ", C[i * N + j]);

  //     printf("\n");
  //   }
  // }

  free(A);
  free(B);
  free(C);

  if (rank == 0)
    free(all_C_blocks);

  MPI_Finalize();

  return 0;
}
