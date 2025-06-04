#ifndef _PHPC_MATRIX_OPERATIONS
#define _PHPC_MATRIX_OPERATIONS

#include <mpi/mpi.h>

int phpc_gemm_sequential(const double *A, int lda, const double *B, int ldb, double *C, int ldc, unsigned int m, unsigned int k, unsigned int n);

void phpc_gemm_summa_cuda(MPI_Comm grid_comm, const double *A, const double *B, double *C, int lda, int ldb, int ldc, int matrices_width, int gpu_count, int grid_width, int grid_height, int block_width);

#endif
