#include <sys/time.h>

#include "utils.h"

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

void initialize_matrix_from_file(const char *file, double **matrix, int *rows, int *cols, int rank) {

  FILE *fp = fopen(file, "rb");
  if (fp == NULL) {
    fprintf(stderr, "Rank %d: Error opening fp for reading: %s\n", rank, file);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  if (fread(rows, sizeof(int), 1, fp) != 1) {
    fprintf(stderr, "Rank %d: Error reading rows from %s\n", rank, file);
    fclose(fp);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  if (fread(cols, sizeof(int), 1, fp) != 1) {
    fprintf(stderr, "Rank %d: Error reading columns from %s\n", rank, file);
    fclose(fp);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  if (*rows <= 0 || *cols <= 0) {
    fprintf(stderr, "Rank %d: Invalid matrix dimensions read from %s: %d x %d\n", rank, file, *rows, *cols);
    fclose(fp);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  size_t num_elements = (size_t)(*rows) * (*cols);
  *matrix = (double *)malloc(num_elements * sizeof(double));
  if (*matrix == NULL) {
    fprintf(stderr, "Rank %d: Memory allocation failed for matrix from %s (%zu elements)\n", rank, file, num_elements);
    fclose(fp);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  if (fread(*matrix, sizeof(double), num_elements, fp) != num_elements) {
    fprintf(stderr, "Rank %d: Error reading matrix data from %s\n", rank, file);
    free(*matrix);
    fclose(fp);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  fclose(fp);
  printf("Successfully read matrix from %s (%d x %d)\n", file, *rows, *cols);
}

void initialize_matrix_to_zero(double **matrix_ptr, int rows, int cols, int rank) {
  if (rows <= 0 || cols <= 0) {
    fprintf(stderr, "Rank %d: Invalid matrix dimensions: %d x %d\n", rank, rows, cols);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  double *matrix = (double *)malloc(rows * cols * sizeof(double));
  if (matrix == NULL) {
    fprintf(stderr, "Rank %d: Memory allocation failed for zero-initialized matrix (%d x %d)\n", rank, rows, cols);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = 0.0;
  }

  *matrix_ptr = matrix;
  printf("Rank %d: Successfully initialized matrix to zero (%d x %d)\n", rank, rows, cols);
}

double get_cur_time() {

  struct timeval tv;
  double cur_time;

  gettimeofday(&tv, NULL);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;

  return cur_time;
}