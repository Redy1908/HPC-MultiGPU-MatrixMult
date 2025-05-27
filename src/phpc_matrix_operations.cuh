#ifndef _PHPC_MATRIX_OPERATIONS
#define _PHPC_MATRIX_OPERATIONS

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>  // sul cluster deve essere #include <mpi.h> in locale se serve mpi/mpi.h

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
int phpc_gemm_cuda(const double *A, const double *B, double *C, unsigned int m, unsigned int k, unsigned int n, dim2 grid_size, unsigned int block_width);
int phpc_gemm_cublas(const double *A, const double *B, double *C, int m, int k, int n);

int phpc_gemm_summa_sequential(const MPI_Comm grid_comm, double *A, double *B, double *C, unsigned int m, unsigned int k, unsigned int n);
int phpc_gemm_summa_cuda(const MPI_Comm grid_comm, double *A, double *B, double *C, unsigned int m, unsigned int k, unsigned int n, dim2 grid_size, unsigned int block_width);
int phpc_gemm_summa_cublas(const MPI_Comm grid_comm, double *A, double *B, double *C, int m, int k, int n);

#if defined(__cplusplus)
}
#endif

#endif
