#!/bin/bash
#SBATCH -p gpus                        # Queue
#SBATCH -N 1                           # Number of nodes (up to 16 nodes * 4 GPUs/node = 64 GPUs)
#SBATCH --ntasks=4                     # Total number of MPI tasks (one per GPU)
#SBATCH --ntasks-per-node=4            # MPI tasks per node (corresponding to GPUs per node)
#SBATCH --gpus-per-node=4              # GPUs per node
#SBATCH --gpus-per-task=1              # GPUs per task
#SBATCH --cpus-per-task=1              # Number of CPU per task
#SBATCH --time=00:05:00                # Maximum time limit 5 min (up to 7 days)
#SBATCH --output=logs/output.log       # Output file
#SBATCH --error=logs/error.log         # Error file
#SBATCH --job-name=matrixmul           # Job name

MPI_INCLUDE_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/include"
MPI_LIB_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/lib64"

ARCH_FLAG=''

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits -i 0)

if [ $? -ne 0 ] || [ -z "$COMPUTE_CAP" ]; then
    echo "Error retrieving compute capability with nvidia-smi or empty output."
    echo "Verify that nvidia-smi is accessible and GPUs are available."
    echo "Using a default architecture (sm_70)."
    ARCH_FLAG="-arch=sm_70"
else
    GPU_ARCH_MAJOR=$(echo $COMPUTE_CAP | cut -d. -f1)
    GPU_ARCH_MINOR=$(echo $COMPUTE_CAP | cut -d. -f2)

    if [ -n "$GPU_ARCH_MAJOR" ] && [ -n "$GPU_ARCH_MINOR" ]; then
        ARCH_FLAG="-arch=sm_${GPU_ARCH_MAJOR}${GPU_ARCH_MINOR}"
    else
        echo "Error parsing compute capability: '$COMPUTE_CAP'."
        echo "Using a default architecture (sm_70)."
        ARCH_FLAG="-arch=sm_70"
    fi
fi

nvcc src/main.cu src/utils.cu src/phpc_matrix_operations.cu -o bin/main_matmul.out -I"$MPI_INCLUDE_PATH" -L"$MPI_LIB_PATH" -Isrc -lcudart -lmpi -lcublas -lm $ARCH_FLAG

srun --mpi=pmix_v3 bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; exec bin/main_matmul.out'