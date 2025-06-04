#ifndef _PHPC_UTILS_H
#define _PHPC_UTILS_H

#include <cuda_runtime.h>
#include <mpi/mpi.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaDeviceProp set_gpu_and_get_properties(int rank);
void check_threads_per_block(cudaDeviceProp prop, int tile_width, int rank);
void check_shared_memory_usage(cudaDeviceProp prop, int tile_width, int rank);

int find_lcm(int a, int b);
double get_cur_time();

#ifdef __cplusplus
}
#endif

#endif