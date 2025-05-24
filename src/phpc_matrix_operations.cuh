#ifndef _PHPC_MATRIX_OPERATIONS
#define _PHPC_MATRIX_OPERATIONS

#include <cuda_runtime.h>
#include <mpi/mpi.h>

typedef struct dim2 {
  unsigned int x, y;

#if defined(__cplusplus)
#if __cplusplus >= 201103L
  __host__ __device__ constexpr dim2(unsigned int vx = 1, unsigned int vy = 1) : x(vx), y(vy) {}
#else
  __host__ __device__ dim2(unsigned int vx = 1, unsigned int vy = 1) : x(vx), y(vy) {}
#endif
#endif
} dim2;

#if defined(__cplusplus)
extern "C" {
#endif

int phpc_gemm_sequential(const double *A, const double *B, double *C, unsigned int m, unsigned int k, unsigned int n);
int phpc_gemm_cuda(const double *A, const double *B, double *C, unsigned int m, unsigned int k, unsigned int n, dim2 grid_size, dim2 block_size);
int phpc_gemm_summa_sequential(const MPI_Comm grid_comm, double *A, double *B, double *C, unsigned int m, unsigned int k, unsigned int n);
int phpc_gemm_summa_cuda(const MPI_Comm grid_comm, double *A, double *B, double *C, unsigned int m, unsigned int k, unsigned int n, dim2 grid_size, dim2 block_size);

#if defined(__cplusplus)
}
#endif

#endif
