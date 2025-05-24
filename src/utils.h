#ifndef _PHPC_UTILS_H
#define _PHPC_UTILS_H

#include <mpi/mpi.h>
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

#define MPI_Assert(check, rank)                                                      \
  if (!(check)) {                                                                    \
    fprintf(stderr, "Error in %s at line %d (Rank %d)\n", __FILE__, __LINE__, rank); \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                         \
  }

int find_lcm(int a, int b);
double get_cur_time();

#ifdef __cplusplus
}
#endif

#endif