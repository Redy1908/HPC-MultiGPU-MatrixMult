#ifndef _PHPC_CUDA_H
#define _PHPC_CUDA_H

#ifdef __cplusplus
extern "C"
#endif
    void
    phpc_gemm_cublas(const double *a, int lda, const double *b, int ldb, double *c, int ldc, int m, int k, int n, int gpu_count, int grid_width, int grid_height, int block_width);

#ifdef __cplusplus
extern "C"
#endif
    void
    phpc_gemm_cuda(const double *a, int lda, const double *b, int ldb, double *c, int ldc, int m, int k, int n, int gpu_count, int grid_width, int grid_height, int block_width);

#ifdef __cplusplus
extern "C"
#endif
    int
    phpc_get_device_count();

#endif
