#!/bin/bash

rm -rf logs/*
rm -rf csv/*
rm -rf bin/*

MPI_INCLUDE_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/include"
MPI_LIB_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/lib64"

nvcc src/main.cu src/utils.cu src/phpc_matrix_operations.cu -o bin/main_matmul.out \
    -I"$MPI_INCLUDE_PATH" -L"$MPI_LIB_PATH" -Isrc \
    -lcudart -lmpi -lcublas -lm -arch=sm_70 -lineinfo

TASK_COUNTS=(1 4 16)
GPU_COUNTS=(1 2 4)
# Add more sizes as needed, size (N) must be divisible by TASK_COUNTS and GPU_COUNTS
# N / TASK_COUNTS = K, N % TASK_COUNTS = 0, K % GPU_COUNTS = 0
MATRIX_SIZES=(256)

for NTASK in "${TASK_COUNTS[@]}"; do
    for NGPU in "${GPU_COUNTS[@]}"; do
        for MSIZE in "${MATRIX_SIZES[@]}"; do

            cat > temp_job_N${MSIZE}_${NTASK}tasks_${NGPU}gpus.slurm << EOF
#!/bin/bash
#SBATCH -p gpus
#SBATCH --ntasks=${NTASK}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=${NGPU}
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/output_N${MSIZE}_${NTASK}tasks_${NGPU}gpus.log
#SBATCH --error=logs/error_N${MSIZE}_${NTASK}tasks_${NGPU}gpus.log
#SBATCH --job-name=mat_mul

srun bin/main_matmul.out ${MSIZE}
EOF
            sbatch temp_job_N${MSIZE}_${NTASK}tasks_${NGPU}gpus.slurm
            
            rm temp_job_N${MSIZE}_${NTASK}tasks_${NGPU}gpus.slurm
        done
    done
done   
