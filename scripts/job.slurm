#!/bin/bash
#SBATCH -p gpus

#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus-per-task=1
#SBATCH --time=00:02:00

#SBATCH hetjob

#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus-per-task=2
#SBATCH --time=00:02:00

mkdir -p bin

MPI_INCLUDE_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/include"
MPI_LIB_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/lib64"

nvcc src/main.cu src/utils.cu src/phpc_matrix_operations.cu -o bin/main.out \
    -I"$MPI_INCLUDE_PATH" -L"$MPI_LIB_PATH" -Isrc \
    -lcudart -lmpi -lcublas -lm -arch=sm_70 -lineinfo

srun --job-name=job0 --output=output0.log --error=error0.log --het-group=0 bin/main.out 512 32 8 8 filename
srun --job-name=job1 --output=output1.log --error=error1.log --het-group=1 bin/main.out 512 32 8 8 filename
