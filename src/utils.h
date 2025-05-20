#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

#define MALLOC_CHECK(ptr, rank, var_name_str)                                           \
  if (ptr == NULL) {                                                                    \
    fprintf(stderr, "MALLOC Error in %s at line %d (Rank %d): Failed to allocate %s\n", \
            __FILE__, __LINE__, rank, var_name_str);                                    \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                            \
  }

#define CUDA_CHECK(err_expr, rank_arg)                                         \
  do {                                                                         \
    cudaError_t err_code = (err_expr);                                         \
    if (err_code != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA Error in %s at line %d (Rank %d): %s\n", __FILE__, \
              __LINE__, rank_arg, cudaGetErrorString(err_code));               \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                 \
    }                                                                          \
  } while (0)

int find_lcm(int a, int b);
void initialize_matrix_from_file(const char *file, double **matrix, int *rows, int *cols, int rank);
void initialize_matrix_to_zero(double **matrix_ptr, int rows, int cols, int rank);
double get_cur_time();

#ifdef __cplusplus
}
#endif

#endif