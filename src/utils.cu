#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include "utils.cuh"

int get_number_of_gpus() {
  int device_count;
  cudaGetDeviceCount(&device_count);

  return device_count;
}

cudaDeviceProp set_gpu_and_get_properties(int rank) {
  cudaDeviceProp prop;
  int device_count, device;

  device_count = get_number_of_gpus();

  if (device_count == 0) {
    fprintf(
        stderr,
        "Rank %d Error in get_gpu_properties: No CUDA-capable devices found.\n",
        rank);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  device = 0;
  cudaSetDevice(device);
  cudaGetDeviceProperties(&prop, device);

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
