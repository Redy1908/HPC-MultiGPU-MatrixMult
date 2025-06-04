#!/bin/bash

MPI_INCLUDE_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/include"
MPI_LIB_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/lib64"

nvcc src/main.cu src/utils.cu src/phpc_matrix_operations.cu -o bin/main_matmul.out \\
    -I"\$MPI_INCLUDE_PATH" -L"\$MPI_LIB_PATH" -Isrc \\
    -lcudart -lmpi -lcublas -lm -arch=sm_70 -lineinfo

PROCESS_COUNTS=(1 4 16)
for NPROCS in "${PROCESS_COUNTS[@]}"; do

    cat > temp_job_${NPROCS}.slurm << EOF
#!/bin/bash
#SBATCH -p gpus
#SBATCH --ntasks=${NPROCS}
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/output_${NPROCS}process.log
#SBATCH --error=logs/error_${NPROCS}process.log
#SBATCH --job-name=matmul_${NPROCS}process

srun --mpi=pmix_v3 bash -c '
    export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
    bin/main_matmul.out
'
EOF

    sbatch temp_job_${NPROCS}.slurm
    
    rm temp_job_${NPROCS}.slurm
done