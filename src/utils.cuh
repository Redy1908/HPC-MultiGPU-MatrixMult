#ifndef _PHPC_UTILS_H
#define _PHPC_UTILS_H

#if defined(__has_include)
#if __has_include(<mpi.h>)
#include <mpi.h>
#else
#include <mpi/mpi.h>
#endif
#endif
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MALLOC_CHECK(ptr, rank, var_name_str)                                           \
  if (ptr == NULL) {                                                                    \
    fprintf(stderr, "MALLOC Error in %s at line %d (Rank %d): Failed to allocate %s\n", \
            __FILE__, __LINE__, rank, var_name_str);                                    \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                            \
  }

#define MPI_Assert(check)                                            \
  if (!(check)) {                                                    \
    fprintf(stderr, "Error in %s at line %d\n", __FILE__, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                         \
  }

#define CUDA_CHECK(err_expr, rank_arg)                                         \
  {                                                                            \
    cudaError_t err_code = (err_expr);                                         \
    if (err_code != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA Error in %s at line %d (Rank %d): %s\n", __FILE__, \
              __LINE__, rank_arg, cudaGetErrorString(err_code));               \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                 \
    }                                                                          \
  }

cudaDeviceProp set_gpu_and_get_properties(int rank);
void check_threads_per_block(cudaDeviceProp prop, int tile_width, int rank);
void check_shared_memory_usage(cudaDeviceProp prop, int tile_width, int rank);

void read_matrix_A_block(const char *filename, double **A, int M, int K, int local_M, int local_K, int proc_row, int lcm, int rank);
void read_matrix_dimensions(const char *filename, int *rows, int *cols, int rank);
void read_matrix_B_block(const char *filename, double **B, int K, int N, int local_K, int local_N, int proc_col, int lcm, int rank);

int get_parameters(int argc, char *const *argv, int *m, int *k, int *n, int *process_grid_dims, int *kernel_grid_size, int *kernel_block_width);

int find_lcm(int a, int b);
double get_cur_time();

#ifdef __cplusplus
}
#endif

#endif