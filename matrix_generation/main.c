#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void generate_matrix(double **matrix_ptr, int rows, int cols, int debug_mode) {
  *matrix_ptr = (double *)malloc(rows * cols * sizeof(double));
  if (*matrix_ptr == NULL) {
    fprintf(stderr, "Memory allocation failed in generate_random_matrix\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (debug_mode) {
        (*matrix_ptr)[i * cols + j] = 2.0;
      } else {
        (*matrix_ptr)[i * cols + j] = (double)rand() / RAND_MAX;
      }
    }
  }
}

void write_matrix_to_file(const char *filename, double *matrix, int rows, int cols) {
  FILE *file = fopen(filename, "wb");
  if (file == NULL) {
    fprintf(stderr, "Error opening file for writing\n");
    exit(EXIT_FAILURE);
  }

  fwrite(&rows, sizeof(int), 1, file);
  fwrite(&cols, sizeof(int), 1, file);
  fwrite(matrix, sizeof(double), rows * cols, file);

  fclose(file);
}

int main() {
  double *A, *B;
  int M_A, K_A, K_B, N_B;
  int debug_mode = 0;

  srand(time(NULL));

  printf("Generate debug matrices (all elements = 2.0 otherwise random values)? (1 for yes, 0 for no): ");
  if (scanf("%d", &debug_mode) != 1) {
    fprintf(stderr, "Invalid input for debug mode selection.\n");
    return EXIT_FAILURE;
  }
  if (debug_mode != 0 && debug_mode != 1) {
    fprintf(stderr, "Debug mode must be 0 or 1.\n");
    return EXIT_FAILURE;
  }

  printf("\nEnter the size of matrix A (MxK):\n");
  printf("Rows (M): ");
  if (scanf("%d", &M_A) != 1 || M_A <= 0) {
    fprintf(stderr, "Invalid input for matrix A rows.\n");
    return EXIT_FAILURE;
  }

  printf("Columns (K): ");
  if (scanf("%d", &K_A) != 1 || K_A <= 0) {
    fprintf(stderr, "Invalid input for matrix A columns.\n");
    return EXIT_FAILURE;
  }

  printf("\nEnter the size of matrix B (KxN):\n");
  printf("Rows (K): ");
  if (scanf("%d", &K_B) != 1 || K_B <= 0) {
    fprintf(stderr, "Invalid input for matrix B rows.\n");
    return EXIT_FAILURE;
  }

  printf("Columns (N): ");
  if (scanf("%d", &N_B) != 1 || N_B <= 0) {
    fprintf(stderr, "Invalid input for matrix B columns.\n");
    return EXIT_FAILURE;
  }

  if (K_A != K_B) {
    fprintf(stderr, "Matrix multiplication not possible: K_A (%d) != K_B (%d)\n", K_A, K_B);
    return EXIT_FAILURE;
  }

  printf("\nGenerating matrix A...\n");
  generate_matrix(&A, M_A, K_A, debug_mode);
  printf("Matrix A generated.\n");

  printf("Generating matrix B...\n");
  generate_matrix(&B, K_B, N_B, debug_mode);
  printf("Matrix B generated.\n");

  printf("\nWriting matrix A to file...\n");
  write_matrix_to_file("inputs/A.bin", A, M_A, K_A);
  printf("Matrix A written to file.\n");

  printf("Writing matrix B to file...\n");
  write_matrix_to_file("inputs/B.bin", B, K_B, N_B);
  printf("Matrix B written to file.\n");

  printf("Matrix A and B generated and written to files.\n");

  free(A);
  free(B);

  return 0;
}