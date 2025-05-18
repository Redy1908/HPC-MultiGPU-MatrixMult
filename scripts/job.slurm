#!/bin/bash
#SBATCH -p gpus
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --output=logs/output.log
#SBATCH --error=logs/error.log

MPI_INCLUDE_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/include"
MPI_LIB_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/lib64"

nvcc -Isrc -I"$MPI_INCLUDE_PATH" -L"$MPI_LIB_PATH" -lcudart -lmpi src/main.cu src/functions.cu src/utils.c -o bin/a.out

mpiexec -np $SLURM_NTASKS bin/a.out
