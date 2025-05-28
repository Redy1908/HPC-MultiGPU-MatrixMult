#include <assert.h>
#include <sys/time.h>
#include <unistd.h>

#include "utils.cuh"

cudaDeviceProp set_gpu_and_get_properties(int rank) {
  cudaDeviceProp prop;
  int device_count, device;

  CUDA_CHECK(cudaGetDeviceCount(&device_count), rank);

  if (device_count == 0) {
    fprintf(
        stderr,
        "Rank %d Error in get_gpu_properties: No CUDA-capable devices found.\n",
        rank);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  device = 0;
  CUDA_CHECK(cudaSetDevice(device), rank);
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device), rank);

  return prop;
}

void check_threads_per_block(cudaDeviceProp prop, int tile_width, int rank) {
  if (rank == 0) {
    int threads_per_block = tile_width * tile_width;

    if (threads_per_block > prop.maxThreadsPerBlock) {
      fprintf(stderr,
              "Rank %d: Error: Threads per block (%d) exceeds GPU max threads per block (%d).\n",
              rank, threads_per_block, prop.maxThreadsPerBlock);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }
}

void check_shared_memory_usage(cudaDeviceProp prop, int tile_width, int rank) {
  if (rank == 0) {
    int threads_per_block = tile_width * tile_width;

    int required_shared_memory_size = 2 * threads_per_block * sizeof(double);

    if (required_shared_memory_size > prop.sharedMemPerBlock) {
      fprintf(stderr,
              "Rank %d: Warning: Required shared memory size (%d bytes) exceeds "
              "available shared memory (%zu bytes) per block. Performance will be affected.\n",
              rank, required_shared_memory_size, prop.sharedMemPerBlock);
    }
  }
}

void read_matrix_dimensions(const char *filename, int *rows, int *cols, int rank) {
  MPI_File file;
  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
  MPI_File_read_at(file, 0, rows, 1, MPI_INT, MPI_STATUS_IGNORE);
  MPI_File_read_at(file, sizeof(int), cols, 1, MPI_INT, MPI_STATUS_IGNORE);
  MPI_File_close(&file);
}

// Legge blocco locale della matrice A (M x K)
// Blocchi: righe in base a dims[0], colonne in base a lcm(dims[0], dims[1])
void read_matrix_A_block(const char *filename, double **A, int M, int K, int local_M, int local_K, int proc_row, int lcm, int rank) {
  MPI_File file;
  MPI_Offset offset;

  *A = (double *)malloc(local_M * local_K * sizeof(double));
  MALLOC_CHECK(*A, rank, "A");

  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);

  // Calcolo offset: righe da proc_row * local_M, colonne da (proc_row * dims[1] + proc_col) * local_K
  // Ma qui leggiamo tutto il blocco locale come blocco contiguo in row-major
  MPI_Offset data_offset = 2 * sizeof(int);  // salto header (dimensioni)
  offset = data_offset + (MPI_Offset)(proc_row * local_M * K) * sizeof(double);

  for (int i = 0; i < local_M; ++i) {
    MPI_File_read_at(file,
                     offset + i * K * sizeof(double),
                     (*A) + i * local_K,
                     local_K,
                     MPI_DOUBLE,
                     MPI_STATUS_IGNORE);
  }

  MPI_File_close(&file);
}

// Legge blocco locale della matrice B (K x N)
// Blocchi: righe in base a lcm(dims[0], dims[1]), colonne in base a dims[1]
void read_matrix_B_block(const char *filename, double **B, int K, int N, int local_K, int local_N, int proc_col, int lcm, int rank) {
  MPI_File file;
  MPI_Datatype filetype;
  int sizes[2] = {K, N};  // dimensione globale
  int subsizes[2] = {local_K, local_N};
  int starts[2] = {0, proc_col * local_N};

  // Ogni processo nella riga del ciclo SUMMA (su K) avanza di un passo
  int proc_row_col_lcm_index = rank % lcm;
  starts[0] = proc_row_col_lcm_index * local_K;

  *B = (double *)malloc(local_K * local_N * sizeof(double));
  MALLOC_CHECK(*B, rank, "B");

  MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &filetype);
  MPI_Type_commit(&filetype);

  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
  MPI_File_set_view(file, 2 * sizeof(int), MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
  MPI_File_read_all(file, *B, local_K * local_N, MPI_DOUBLE, MPI_STATUS_IGNORE);

  MPI_File_close(&file);
  MPI_Type_free(&filetype);
}

int get_parameters(int argc, char *const *argv, int *process_grid_dims, int *kernel_grid_size, int *kernel_block_width) {
  if (process_grid_dims == NULL || kernel_grid_size == NULL || kernel_block_width == NULL)
    return 1;

  /* initialize to some invalid values so it returns error if any of them is not set correctly */
  process_grid_dims[0] = process_grid_dims[1] = 0;
  kernel_grid_size[0] = kernel_grid_size[1] = 0;
  *kernel_block_width = 0;

  int c;
  while ((c = getopt(argc, argv, "r:c:w:x:y:")) != -1) {
    switch (c) {
      case 'r':
        process_grid_dims[0] = atoi(optarg);
        break;
      case 'c':
        process_grid_dims[1] = atoi(optarg);
        break;
      case 'w':
        *kernel_block_width = atoi(optarg);
        break;
      case 'x':
        kernel_grid_size[0] = atoi(optarg);
        break;
      case 'y':
        kernel_grid_size[1] = atoi(optarg);
        break;
      default:
        break;
    }
  }

  if (process_grid_dims[0] <= 0 || process_grid_dims[1] <= 0 || kernel_grid_size[0] <= 0 || kernel_grid_size[1] <= 0 || *kernel_block_width <= 0)
    return 2;

  return 0;
}

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
