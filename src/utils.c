#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>

#include "utils.h"

int initialize_matrix_from_file(const char* file, double** matrix, int* rows, int* cols) {

    FILE* fp = fopen(file, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error opening fp for reading: %s\n", file);
        *matrix = NULL;
        return -1;
    }

    if (fread(rows, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Error reading rows from %s\n", file);
        fclose(fp);
        *matrix = NULL;
        return -1;
    }

    if (fread(cols, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Error reading columns from %s\n", file);
        fclose(fp);
        *matrix = NULL;
        return -1;
    }

    if (*rows <= 0 || *cols <= 0) {
        fprintf(stderr, "Invalid matrix dimensions read from %s: %d x %d\n", file, *rows, *cols);
        fclose(fp);
        *matrix = NULL;
        return -1;
    }

    size_t num_elements = (size_t)(*rows) * (*cols);
    *matrix = (double*)malloc(num_elements * sizeof(double));
    if (*matrix == NULL) {
        fprintf(stderr, "Memory allocation failed for matrix from %s (%zu elements)\n", file, num_elements);
        fclose(fp);
        return -1;
    }

    if (fread(*matrix, sizeof(double), num_elements, fp) != num_elements) {
        fprintf(stderr, "Error reading matrix data from %s\n", file);
        free(*matrix);
        *matrix = NULL;
        fclose(fp);
        return -1;
    }

    fclose(fp);
    printf("Successfully read matrix from %s (%d x %d)\n", file, *rows, *cols);
    return 0;
}

int initialize_matrix_to_zero(double** matrix_ptr, int rows, int cols){
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "Invalid matrix dimensions: %d x %d\n", rows, cols);
        return -1;
    }

    double* matrix = (double*)malloc(rows * cols * sizeof(double));
    if (matrix == NULL) {
        fprintf(stderr, "Memory allocation failed for zero-initialized matrix (%d x %d)\n", rows, cols);
        return -1;
    }

    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = 0.0;
    }

    *matrix_ptr = matrix;
    printf("Successfully initialized matrix to zero (%d x %d)\n", rows, cols);

    return 0;
}

double get_cur_time() {
    struct timespec ts;
    
    clock_gettime(CLOCK_MONOTONIC, &ts);
    
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}