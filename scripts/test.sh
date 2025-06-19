#!/bin/bash

mkdir -p bin

MPI_INCLUDE_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/include"
MPI_LIB_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/lib64"

nvcc src/main.cu -o bin/main.out -I"$MPI_INCLUDE_PATH" -L"$MPI_LIB_PATH" -lmpi && sbatch $1
