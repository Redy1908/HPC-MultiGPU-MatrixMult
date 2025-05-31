#!/bin/bash
#SBATCH -p gpus                        # Queue
#SBATCH --ntasks=4                     # Total number of MPI tasks
#SBATCH --gpus-per-task=1              # GPUs per task
#SBATCH --time=00:05:00                # Maximum time limit 5 min (up to 7 days)
#SBATCH --output=logs/output.log       # Output file
#SBATCH --error=logs/error.log         # Error file
#SBATCH --job-name=matrixmul           # Job name

MPI_INCLUDE_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/include"
MPI_LIB_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/lib64"

nvcc src/main.cu src/utils.cu src/phpc_matrix_operations.cu -o bin/main_matmul.out \
    -I"$MPI_INCLUDE_PATH" -L"$MPI_LIB_PATH" -Isrc \
    -lcudart -lmpi -lcublas -lm -arch=sm_70 \
    -lineinfo


export OMPI_MCA_pml=ucx

#multi nodo al momento non funziona dc_mlx5,rc_mlx5 dovrebbero essere per la comunicazione tra nodi con infiniband
#export UCX_TLS=dc_mlx5,rc_mlx5,cuda_ipc,cuda_copy,gdr_copy,sm,self

#singolo nodo funziona
export UCX_TLS=cuda_ipc,cuda_copy,gdr_copy,sm,self

srun --mpi=pmix_v3 bash -c '
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    nsys profile \
        --output=profiling/nsys_profile_rank_${SLURM_PROCID} \
        --force-overwrite true \
        --gpu-metrics-device=all \
        --cuda-memory-usage=true \
        --stats=false \
        --trace=cuda,cublas,mpi,nvtx \
        bin/main_matmul.out'