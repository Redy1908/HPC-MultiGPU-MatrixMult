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

JOBS_IDS=()

echo "Generating and submitting SLURM jobs based on CSV files in the 'tests' directory..."
CSV_FILES_DIR="tests"

for csv_file_path in "$CSV_FILES_DIR"/*.csv; do
    if [ ! -f "$csv_file_path" ]; then
        echo "Warning: No CSV files found in $CSV_FILES_DIR, or $csv_file_path is not a file."
        continue
    fi

    csv_filename=$(basename "$csv_file_path")
    csv_filename_no_ext="${csv_filename%.*}"

    {
        read header
        while IFS= read -r line || [[ -n "$line" ]]; do
            [[ -z "$line" ]] && continue
            
            IFS=',' read -r matrix_width processes GPU_number tile_width grid_width grid_height <<< "$line"

            MSIZE=$(echo "$matrix_width" | xargs)
            NTASK=$(echo "$processes" | xargs)
            NGPU=$(echo "$GPU_number" | xargs)
            TILE_WIDTH=$(echo "$tile_width" | xargs)
            GRID_WIDTH=$(echo "$grid_width" | xargs)
            GRID_HEIGHT=$(echo "$grid_height" | xargs)
            
            JOB_NAME_SUFFIX="${csv_filename_no_ext}_N${MSIZE}_T${NTASK}_G${NGPU}_TW${TILE_WIDTH}"
            SLURM_SCRIPT_NAME_TMP="job_${JOB_NAME_SUFFIX}.slurm"

            cat > "${SLURM_SCRIPT_NAME_TMP}" << EOF
#!/bin/bash
#SBATCH -p gpus
#SBATCH --ntasks=${NTASK}
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus-per-task=${NGPU}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu="5G"
#SBATCH --time=00:20:00
#SBATCH --output=logs/output_${JOB_NAME_SUFFIX}.log
#SBATCH --error=logs/error_${JOB_NAME_SUFFIX}.log
#SBATCH --job-name=mat_mul

srun nsys profile \
    --force-overwrite true \
    --gpu-metrics-device=all \
    --output=profiling/profile_${JOB_NAME_SUFFIX}_procid\$SLURM_PROCID \
    bin/main.out ${MSIZE} ${TILE_WIDTH} ${GRID_WIDTH} ${GRID_HEIGHT} ${csv_filename_no_ext}
EOF
            JOB_OUTPUT=$(sbatch ${SLURM_SCRIPT_NAME_TMP})
            JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]*$')
            if [ -n "$JOB_ID" ]; then
                echo "    Submitting SLURM script: ${SLURM_SCRIPT_NAME_TMP} with Job ID: ${JOB_ID}"
                JOBS_IDS+=("$JOB_ID")
            else
                echo "    Error submitting SLURM script: ${SLURM_SCRIPT_NAME_TMP}. sbatch output: $JOB_OUTPUT"
            fi
                
            rm "${SLURM_SCRIPT_NAME_TMP}"
        done 
    } < "$csv_file_path"
done

echo " "
if [ ${#JOBS_IDS[@]} -eq 0 ]; then
    echo "No SLURM jobs were submitted."
    echo "Skipping plot job submission."
else
    echo "All SLURM jobs submitted."

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

    JOB_OUTPUT_PLOT=$(sbatch plot_job.slurm)
    JOB_ID_PLOT=$(echo "$JOB_OUTPUT_PLOT" | grep -o '[0-9]*$')
    if [ -n "$JOB_ID_PLOT" ]; then
        echo "Submitting SLURM script: plot_job.slurm with Job ID: ${JOB_ID_PLOT}"
    else
        echo "Error submitting plot_job.slurm. sbatch output: $JOB_OUTPUT_PLOT"
    fi
    echo " "

    rm plot_job.slurm
fi