#!/bin/bash
#SBATCH -p gpus                        # Queue
#SBATCH --nodes=4                    # Total number of MPI tasks
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1              # GPUs per task
#SBATCH --time=00:05:00                # Maximum time limit 5 min (up to 7 days)
#SBATCH --output=output.log       # Output file
#SBATCH --error=error.log         # Error file
#SBATCH --job-name=nodes-test           # Job name

# mpicc ./main.c -o ./main && srun ./main

MPI_INCLUDE_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/include"
MPI_LIB_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/lib64"

# nvcc ./main.cu -o ./main -I"$MPI_INCLUDE_PATH" -L"$MPI_LIB_PATH" -lmpi && srun ./main

nvcc src/main.cu src/utils.cu src/phpc_matrix_operations.cu -o ./main -I"$MPI_INCLUDE_PATH" -L"$MPI_LIB_PATH" -lmpi -lcublas && mpirun -n 6 ./main
