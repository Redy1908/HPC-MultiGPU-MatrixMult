#ifndef _PHPC_UTILS_H
#define _PHPC_UTILS_H

#include <stdio.h>

int find_lcm(int a, int b);
double get_cur_time();
void log_to_csv(FILE *csv_file, int N, int size, int num_blocks, int threads_per_block, const char *method, double time);

#endif