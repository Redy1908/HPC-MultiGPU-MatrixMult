#!/bin/bash

for dir in logs csv plots profiling bin; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"/*
    else
        mkdir -p "$dir"
    fi
done

MPI_INCLUDE_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/include"
MPI_LIB_PATH="/usr/mpi/gcc/openmpi-4.1.0rc5/lib64"

echo " "
echo "Compiling main.out..."

nvcc src/main.cu src/utils.cu src/phpc_matrix_operations.cu -o bin/main.out \
    -I"$MPI_INCLUDE_PATH" -L"$MPI_LIB_PATH" -Isrc \
    -lcudart -lmpi -lcublas -lm -arch=sm_70 -lineinfo

if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting..."
    exit 1
fi

echo "Compilation successful."
echo " "

TASK_COUNTS=(1 4 16)
GPU_COUNTS=(1 2 4)
MATRIX_SIZES=(256)

# ensure 1 is always present since it is used for the calculation of baseline
TILE_WIDTH=(1 4 8 16 32)

# Array to hold job IDs used for crating job dependencies for the analysis job
JOBS_IDS=()

echo "Starting submission of SLURM jobs..."
for NTASK in "${TASK_COUNTS[@]}"; do
    for NGPU in "${GPU_COUNTS[@]}"; do
        for MSIZE in "${MATRIX_SIZES[@]}"; do

            SLURM_SCRIPT="job_N${MSIZE}_${NTASK}tasks_${NGPU}gpus.slurm"

            cat > ${SLURM_SCRIPT} << EOF
#!/bin/bash
#SBATCH -p gpus
#SBATCH --ntasks=${NTASK}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=${NGPU}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu="5G"
#SBATCH --time=00:20:00
#SBATCH --output=logs/output_N${MSIZE}_${NTASK}tasks_${NGPU}gpus.log
#SBATCH --error=logs/error_N${MSIZE}_${NTASK}tasks_${NGPU}gpus.log
#SBATCH --job-name=mat_mul

srun nsys profile \
    --force-overwrite true \
    --gpu-metrics-device=all \
    --output=profiling/profile_N${MSIZE}_${NTASK}tasks_${NGPU}gpus_procid\$SLURM_PROCID \
    bin/main.out ${MSIZE} ${#TILE_WIDTH[@]} ${TILE_WIDTH[*]}
EOF
            JOB_OUTPUT=$(sbatch ${SLURM_SCRIPT})
            JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]*$')
            echo "Submitting SLURM script: ${SLURM_SCRIPT} with Job ID: ${JOB_ID}"

            JOBS_IDS+=("$JOB_ID")
            
            rm ${SLURM_SCRIPT}
        done
    done
done

echo " "
echo "All SLURM jobs submitted."

# Create a job dependency string with all job IDs the analysis job waits for all jobs to create their output (csv files)
DEPENDENCY_STRING=$(IFS=:; echo "${JOBS_IDS[*]}")

cat > plot_job.slurm << EOF
#!/bin/bash
#SBATCH -p sequential
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem="1G"
#SBATCH --time=00:10:00
#SBATCH --output=logs/analysis_output.log
#SBATCH --error=logs/analysis_error.log
#SBATCH --job-name=analysis
#SBATCH --dependency=afterany:${DEPENDENCY_STRING}

source /nfsexports/SOFTWARE/anaconda3.OK/etc/profile.d/conda.sh
conda activate base

python3 scripts/plots.py
EOF

echo " "
echo "Plot job depending on all previous jobs..."

JOB_OUTPUT=$(sbatch plot_job.slurm)
JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]*$')
echo "Submitting SLURM script: plot_job.slurm with Job ID: ${JOB_ID}"
echo " "

rm plot_job.slurm