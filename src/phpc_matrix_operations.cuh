#ifndef _PHPC_MATRIX_OPERATIONS
#define _PHPC_MATRIX_OPERATIONS

#include <mpi.h>

#if defined(__cplusplus)
extern "C" {
#endif

void phpc_gemm_summa_cuda(MPI_Comm grid_comm, double *A, double *B, double *C, int ld, int M, int K, int N, dim3 dim_block, dim3 dim_grid, int shared_mem_size);

#if defined(__cplusplus)
}
#endif

#endif
