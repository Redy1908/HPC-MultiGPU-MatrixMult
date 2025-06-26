#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "utils.cuh"

int find_lcm(int a, int b) {
  int q, r;
  int x = a;
  int y = b;

  while (y != 0) {
    q = x / y;
    r = x - q * y;
    x = y;
    y = r;
  }

  return a * b / x;
}

double get_cur_time() {
  struct timeval tv;
  struct timezone tz;
  double cur_time;

  gettimeofday(&tv, &tz);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;

  return cur_time;
}

void log_to_csv(FILE *csv_file, int N, int size, int gpu_count, int num_blocks, int threads_per_block, const char *method, double time) {
  if (csv_file) {
    if (strcmp(method, "ITERATIVE") == 0) {
      fprintf(csv_file, "%d,%d,%d,%d,%d,%s,%f\n", N, 1, 0, 0, 0, method, time);
    } else if (strcmp(method, "SUMMA_CUBLAS") == 0) {
      fprintf(csv_file, "%d,%d,%d,%d,%d,%s,%f\n", N, size, gpu_count, 0, 0, method, time);
    } else {
      fprintf(csv_file, "%d,%d,%d,%d,%d,%s,%f\n", N, size, gpu_count, num_blocks, threads_per_block, method, time);
    }
  }
}