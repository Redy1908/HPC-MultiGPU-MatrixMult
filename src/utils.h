#ifndef _PHPC_UTILS_H
#define _PHPC_UTILS_H

#include <stdio.h>

double get_cur_time();
void log_to_csv(FILE *csv_file, int N, int size, int gpu_count, int num_blocks, int threads_per_block, double cuda_time, float cuda_gpu_time, double cublas_time);

#endif