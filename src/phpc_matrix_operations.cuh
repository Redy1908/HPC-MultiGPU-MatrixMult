#ifndef _PHPC_MATRIX_OPERATIONS
#define _PHPC_MATRIX_OPERATIONS

#include <mpi/mpi.h>

#ifdef __cplusplus
extern "C"
#endif
    void
    phpc_gemm_iterative(const double *A, const double *B, double *C, int N);

#ifdef __cplusplus
extern "C"
#endif
    void
    phpc_gemm_summa_cuda(MPI_Comm grid_comm, const double *A, const double *B, double *C, int N, int gpu_count, int grid_width, int grid_height, int block_width);

#ifdef __cplusplus
extern "C"
#endif
    void
    phpc_gemm_summa_cublas(MPI_Comm grid_comm, const double *A, const double *B, double *C, int N, int gpu_count);

#endif
