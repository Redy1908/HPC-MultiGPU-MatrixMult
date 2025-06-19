#!/bin/bash
#SBATCH -p gpus
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/output.log
#SBATCH --error=logs/error.log
#SBATCH --job-name=script-testing

mkdir -p bin

MPI_INCLUDE_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/include"
MPI_LIB_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/lib64"

nvcc src/main.cu src/utils.cu src/phpc_matrix_operations.cu -o bin/main.out \
    -I"$MPI_INCLUDE_PATH" -L"$MPI_LIB_PATH" -Isrc \
    -lcudart -lmpi -lcublas -lm -arch=sm_70 -lineinfo

srun bin/main.out 2048 32 8 8 filename
