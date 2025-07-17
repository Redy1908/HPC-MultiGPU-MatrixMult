#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

double get_cur_time() {
  struct timeval tv;
  // struct timezone tz;
  double cur_time;

  gettimeofday(&tv, NULL);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;

  return cur_time;
}

void log_to_csv(FILE *csv_file, int N, int size, int gpu_count, int num_blocks, int threads_per_block, double iterative_time, double cuda_time, double cublas_time) {
  // if (csv_file) {
  //   if (strcmp(method, "ITERATIVE") == 0) {
  //     fprintf(csv_file, "%d,%d,%d,%d,%d,%s,%f\n", N, 1, 0, 0, 0, method, time);
  //   } else if (strcmp(method, "SUMMA_CUBLAS") == 0) {
  //     fprintf(csv_file, "%d,%d,%d,%d,%d,%s,%f\n", N, size, gpu_count, 0, 0, method, time);
  //   } else {
  int total_threads = gpu_count * num_blocks * threads_per_block;
  fprintf(csv_file, "%d,%d,%d,%d,%d,%d,%f,%f,%f\n", N, size, gpu_count, num_blocks, threads_per_block, total_threads, iterative_time, cuda_time, cublas_time);
  // }
  // }
}