#ifndef _PHPC_SUMMA_H
#define _PHPC_SUMMA_H

#include <mpi/mpi.h>

void phpc_gemm_summa_cuda(MPI_Comm grid_comm, const double *A, const double *B, double *C, int n, int gpu_count, int grid_width, int grid_height, int block_width, float *compute_time);

void phpc_gemm_summa_cublas(MPI_Comm grid_comm, const double *A, const double *B, double *C, int n, int gpu_count, float *compute_time);

#endif
