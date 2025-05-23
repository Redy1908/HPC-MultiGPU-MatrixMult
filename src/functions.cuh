#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUDA_CHECK(err_expr, rank_arg)                                         \
  do {                                                                         \
    cudaError_t err_code = (err_expr);                                         \
    if (err_code != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA Error in %s at line %d (Rank %d): %s\n", __FILE__, \
              __LINE__, rank_arg, cudaGetErrorString(err_code));               \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                 \
    }                                                                          \
  } while (0)

cudaDeviceProp set_gpu_and_get_properties(int rank);
void check_threads_per_block(cudaDeviceProp prop, int tile_width, int rank);
void check_shared_memory_usage(cudaDeviceProp prop, int tile_width, int rank);
void SUMMA(MPI_Comm grid_comm, double *A, double *B, double *C, int M, int K, int N, int tile_width, int rank);
void read_matrix_A_block(const char *filename, double **A, int M, int K, int local_M, int local_K, int proc_row, int lcm, int rank);
void read_matrix_dimensions(const char *filename, int *rows, int *cols, int rank);
void read_matrix_B_block(const char *filename, double **B, int K, int N, int local_K, int local_N, int proc_col, int lcm, int rank);
__global__ void matrix_mul_kernel(double *A, double *B, double *C, int M, int N, int K);

#ifdef __cplusplus
}
#endif

#endif
