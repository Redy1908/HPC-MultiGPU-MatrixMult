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

for NTASK in "${TASK_COUNTS[@]}"; do

    cat > temp_job_${NTASK}.slurm << EOF
#!/bin/bash
#SBATCH -p gpus
#SBATCH --ntasks=${NTASK}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/output_${NTASK}process.log
#SBATCH --error=logs/error_${NTASK}process.log
#SBATCH --job-name=matmul_${NTASK}process

srun bin/main_matmul.out
EOF

    sbatch temp_job_${NTASK}.slurm
    
    rm temp_job_${NTASK}.slurm
done