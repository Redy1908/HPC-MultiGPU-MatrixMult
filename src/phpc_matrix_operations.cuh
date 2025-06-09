#ifndef _PHPC_MATRIX_OPERATIONS
#define _PHPC_MATRIX_OPERATIONS

#include <mpi.h>

int phpc_gemm_summa_cuda(MPI_Comm grid_comm, const double *A, const double *B, double *C, int N, int gpu_count, int grid_width, int grid_height, int block_width);

#endif
