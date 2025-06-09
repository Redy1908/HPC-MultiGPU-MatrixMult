#ifndef _PHPC_UTILS_H
#define _PHPC_UTILS_H

#include <cuda_runtime.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

int find_lcm(int a, int b);
double get_cur_time();

#ifdef __cplusplus
}
#endif

#endif