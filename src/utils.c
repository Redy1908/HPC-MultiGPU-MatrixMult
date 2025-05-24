#include "utils.h"

#include <sys/time.h>

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
  double cur_time;

  gettimeofday(&tv, NULL);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;

  return cur_time;
}
