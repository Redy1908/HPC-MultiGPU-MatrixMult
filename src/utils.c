#include <time.h>
#include <unistd.h>

#include "utils.h"

double get_cur_time() {
    struct timespec ts;
    
    clock_gettime(CLOCK_MONOTONIC, &ts);
    
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}