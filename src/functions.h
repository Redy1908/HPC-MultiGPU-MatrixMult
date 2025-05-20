#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#ifdef __cplusplus
extern "C" {
#endif

cudaDeviceProp set_gpu_and_get_properties(int rank);
void check_threads_per_block(cudaDeviceProp prop, int tile_width, int rank);
void check_shared_memory_usage(cudaDeviceProp prop, int tile_width, int rank);
void SUMMA(MPI_Comm grid_comm, double *A, double *B, double *C, int M, int K, int N, int tile_width, int rank);
__global__ void matrix_mul_kernel(double *A, double *B, double *C, int M, int N, int K);

#ifdef __cplusplus
}
#endif

#endif