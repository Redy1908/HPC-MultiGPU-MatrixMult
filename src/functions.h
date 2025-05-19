#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#ifdef __cplusplus
extern "C" {
#endif

cudaDeviceProp set_gpu_and_get_properties(int rank);
int calculate_optimal_tile_width(cudaDeviceProp prop, int rank);
__global__ void matrix_mul_kernel(double* A, double* B, double* C, int M, int N, int K);

#ifdef __cplusplus
}
#endif

#endif 