#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

int initialize_matrix_from_file(const char* file, double** matrix, int* rows, int* cols);
int initialize_matrix_to_zero(double** matrix_ptr, int rows, int cols);
double get_cur_time();

#ifdef __cplusplus
}
#endif

#endif 