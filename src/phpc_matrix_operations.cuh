#ifndef _PHPC_MATRIX_OPERATIONS
#define _PHPC_MATRIX_OPERATIONS

#ifdef __cplusplus
extern "C"
#endif
    void
    phpc_gemm_cuda(const double *a, int lda, const double *b, int ldb, double *c, int ldc, int m, int k, int n, int gpu_count, int grid_width, int grid_height, int block_width);

#endif
