#!/bin/bash

mkdir -p bin

# MPI_INCLUDE_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/include"
# MPI_LIB_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/lib64"

nvcc src/main.cu -o bin/main.out -I/usr/mpi/gcc/openmpi-4.1.0rc5/include -pthread -L/usr/lib64 -Wl,-rpath -Wl,/usr/lib64 -Wl,-rpath -Wl,/usr/mpi/gcc/openmpi-4.1.0rc5/lib64 -Wl,--enable-new-dtags -L/usr/mpi/gcc/openmpi-4.1.0rc5/lib64 -lmpi
