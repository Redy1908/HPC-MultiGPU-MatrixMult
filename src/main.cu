#include <mpi.h>
#include <stdio.h>

#include "functions.h"
#include "utils.h"

#define MALLOC_CHECK(ptr, rank_main, var_name_str)                                     \
  if (ptr == NULL) {                                                                   \
    fprintf(stderr, "MALLOC Error in %s at line %d (Rank %d): Failed to allocate %s\n",\
            __FILE__, __LINE__, rank_main, var_name_str);                              \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                           \
  }

#define CUDA_CHECK(err, rank)                                                         \
  if (err != cudaSuccess) {                                                           \
    fprintf(stderr, "CUDA Error in %s at line %d (Rank %d): %s\n", __FILE__, __LINE__,\
            rank, cudaGetErrorString(err));                                           \
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                          \
  }

int get_number_of_gpus() {
  int device_count;
  cudaGetDeviceCount(&device_count);

  return device_count;
}

int main(int argc, char *argv[]) {

  int size, rank;
  double *h_A_full = NULL, *h_B_full = NULL, *h_C_full = NULL; 
  double *h_A_proc = NULL, *h_B_proc = NULL, *h_C_proc = NULL;         
  double *d_A_proc = NULL, *d_B_proc = NULL, *d_C_proc = NULL;  
  int M_A, K_A, K_B, N_B;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == 0) {
    printf("Process 0: reading matix A from file...\n");
    initialize_matrix_from_file("inputs/A.bin", &h_A_full, &M_A, &K_A, rank);

    printf("\nProcess 0: reading matix B from file...\n");
    initialize_matrix_from_file("inputs/B.bin", &h_B_full, &K_B, &N_B, rank);

    if (K_A != K_B) {
      fprintf(stderr, "Rank %d: Error: Matrix A's columns (%d) must match Matrix B's rows (%d).\n", rank, K_A, K_B);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    printf("\nProcess 0: allocatiing and initializing matrix C to 0...\n");
    initialize_matrix_to_zero(&h_C_full, M_A, N_B, rank);
  }

  MPI_Bcast(&M_A, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&K_A, 1, MPI_INT, 0, MPI_COMM_WORLD); 
  MPI_Bcast(&N_B, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Number of rows of A for each MPI process.
  int *rows_per_proc = (int*)malloc(size * sizeof(int));
  MALLOC_CHECK(rows_per_proc, rank, "rows_per_proc");

  // Starting row index of A for each process.
  int *displs_rows_start = (int*)malloc(size * sizeof(int));
  MALLOC_CHECK(displs_rows_start, rank, "displs_rows_start"); 
  
  // Number of elements of A to send to each process.
  int *sendcounts_elements_A = (int*)malloc(size * sizeof(int));
  MALLOC_CHECK(sendcounts_elements_A, rank, "sendcounts_elements_A");
  
  // Displacement for MPI_Scatterv of A elements.
  int *displs_elements_A = (int*)malloc(size * sizeof(int));
  MALLOC_CHECK(displs_elements_A, rank, "displs_elements_A");    
  
  // Number of elements of C to receive from each process.
  int *recvcounts_elements_C = (int*)malloc(size * sizeof(int));
  MALLOC_CHECK(recvcounts_elements_C, rank, "recvcounts_elements_C");
  
  // Displacement for MPI_Gatherv of C elements.
  int *displs_elements_C = (int*)malloc(size * sizeof(int));
  MALLOC_CHECK(displs_elements_C, rank, "displs_elements_C");

  int base_rows = M_A / size;
  int remaining_rows = M_A % size;
  int current_row_offset = 0;
  int current_element_offset_A = 0;
  int current_element_offset_C = 0;

  for(int i = 0; i < size; i++){
    rows_per_proc[i] = base_rows + (i < remaining_rows ? 1 : 0);
    displs_rows_start[i] = current_row_offset;

    sendcounts_elements_A[i] = rows_per_proc[i] * K_A;
    displs_elements_A[i] = current_element_offset_A;

    recvcounts_elements_C[i] = rows_per_proc[i] * N_B;
    displs_elements_C[i] = current_element_offset_C;

    current_row_offset += rows_per_proc[i];
    current_element_offset_A += sendcounts_elements_A[i];
    current_element_offset_C += recvcounts_elements_C[i];
  }

  int M_proc = rows_per_proc[rank];

  if(M_proc > 0){
    h_A_proc = (double*)malloc(M_proc * K_A * sizeof(double));
    MALLOC_CHECK(h_A_proc, rank, "h_A_proc");

    h_C_proc = (double*)malloc(M_proc * N_B * sizeof(double));
    MALLOC_CHECK(h_C_proc, rank, "h_C_proc");
  }
  h_B_proc = (double*)malloc(K_A * N_B * sizeof(double));
  MALLOC_CHECK(h_B_proc, rank, "h_B_proc");

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Scatterv(h_A_full, sendcounts_elements_A, displs_elements_A, MPI_DOUBLE, 
               h_A_proc, sendcounts_elements_A[rank], MPI_DOUBLE, 
               0, MPI_COMM_WORLD);
             
  if (rank == 0) {
    memcpy(h_B_proc, h_B_full, (size_t)K_A * N_B * sizeof(double));
  }
  MPI_Bcast(h_B_proc, K_A * N_B, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (M_proc > 0) { 
    CUDA_CHECK(cudaMalloc((void**)&d_A_proc, (size_t)M_proc * K_A * sizeof(double)), rank);
    CUDA_CHECK(cudaMalloc((void**)&d_C_proc, (size_t)M_proc * N_B * sizeof(double)), rank);
    CUDA_CHECK(cudaMemcpy(d_A_proc, h_A_proc, (size_t)M_proc * K_A * sizeof(double), cudaMemcpyHostToDevice), rank);

    CUDA_CHECK(cudaMalloc((void**)&d_B_proc, (size_t)K_A * N_B * sizeof(double)), rank);
    CUDA_CHECK(cudaMemcpy(d_B_proc, h_B_proc, (size_t)K_A * N_B * sizeof(double), cudaMemcpyHostToDevice), rank);

    cudaDeviceProp gpu_prop = set_gpu_and_get_properties(rank);
    int tile_width = calculate_optimal_tile_width(gpu_prop, rank);
    size_t total_shared_memory_bytes = 2 * tile_width * tile_width * sizeof(double);

    dim3 dimGrid(ceil(N_B / (float)tile_width), ceil(M_proc/ (float)tile_width), 1);
    dim3 dimBlock(tile_width, tile_width, 1);

    matrix_mul_kernel<<<dimGrid, dimBlock, total_shared_memory_bytes>>>(d_A_proc, d_B_proc, d_C_proc, M_proc, N_B, K_A);

    CUDA_CHECK(cudaGetLastError(), rank); 
    CUDA_CHECK(cudaDeviceSynchronize(), rank);

    CUDA_CHECK(cudaMemcpy(h_C_proc, d_C_proc, (size_t)M_proc * N_B * sizeof(double), cudaMemcpyDeviceToHost), rank);
  }

  MPI_Gatherv(h_C_proc, recvcounts_elements_C[rank], MPI_DOUBLE, 
              h_C_full, recvcounts_elements_C, displs_elements_C, MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  // h_C_full is now the result of the matrix multiplication

  // -----------------------------------------------------------------------------------------
  // Cleaning 
  // -----------------------------------------------------------------------------------------

  if(M_proc > 0){
    CUDA_CHECK(cudaFree(d_A_proc), rank);
    CUDA_CHECK(cudaFree(d_B_proc), rank);
    CUDA_CHECK(cudaFree(d_C_proc), rank);
  }

  free(h_A_proc); 
  free(h_B_proc);
  free(h_C_proc); 

  if (rank == 0) {
    free(h_A_full);
    free(h_B_full);
    free(h_C_full);
  }

  free(rows_per_proc);
  free(displs_rows_start);
  free(sendcounts_elements_A);
  free(displs_elements_A);
  free(recvcounts_elements_C);
  free(displs_elements_C);

  MPI_Finalize();
  return 0;
}
