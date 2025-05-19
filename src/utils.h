#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

void initialize_matrix_from_file(const char* file, double** matrix, int* rows, int* cols, int rank);
void initialize_matrix_to_zero(double** matrix_ptr, int rows, int cols, int rank);
double get_cur_time();

#ifdef __cplusplus
}
#endif

#endif 