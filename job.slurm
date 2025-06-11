#!/bin/bash
#SBATCH -p gpus
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --job-name=gpu-testing

MPI_INCLUDE_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/include"
MPI_LIB_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/lib64"

nvcc src/tests/multigpu.cu src/utils.cu src/phpc_matrix_operations.cu -o main.out -I"$MPI_INCLUDE_PATH" -L"$MPI_LIB_PATH" -lmpi -lcublas && srun main.out 32 2 2

wait
