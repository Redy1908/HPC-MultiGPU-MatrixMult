#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>   

#ifdef __cplusplus
extern "C" {
#endif

#define MALLOC_CHECK(ptr, rank, var_name_str)                                           \
  if (ptr == NULL) {                                                                    \
    fprintf(stderr, "MALLOC Error in %s at line %d (Rank %d): Failed to allocate %s\n", \
            __FILE__, __LINE__, rank, var_name_str);                                    \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                            \
  }

int find_lcm(int a, int b);
void initialize_matrix_from_file(const char *file, double **matrix, int *rows, int *cols, int rank);
void initialize_matrix_to_zero(double **matrix_ptr, int rows, int cols, int rank);
double get_cur_time();

#ifdef __cplusplus
}
#endif

#endif