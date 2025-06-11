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

# Array to hold job IDs used for creating job dependencies for the analysis job
JOBS_IDS=()

echo "Starting submission of SLURM jobs based on CSV files in tests/ ..."
CSV_FILES_DIR="tests"

for csv_file_path in "$CSV_FILES_DIR"/*.csv; do
    if [ ! -f "$csv_file_path" ]; then
        echo "Warning: No CSV files found in $CSV_FILES_DIR, or $csv_file_path is not a file."
        continue # Skip if not a file (e.g. if glob doesn't match anything)
    fi

    csv_filename=$(basename "$csv_file_path")
    csv_filename_no_ext="${csv_filename%.*}"
    echo "Processing CSV file: $csv_filename"

    # Read CSV, skip header (tail -n +2), process data lines
    row_num=0
    tail -n +2 "$csv_file_path" | while IFS=, read -r matrix_width_csv processes_csv GPU_number_csv gpu_blocks_csv gpu_threads_csv || [[ -n "$matrix_width_csv" ]]; do
        # Handle potential empty last line if not properly terminated or empty lines
        if [ -z "$matrix_width_csv" ]; then continue; fi
        row_num=$((row_num + 1))

        # Trim whitespace
        MSIZE=$(echo "$matrix_width_csv" | xargs)
        NTASK=$(echo "$processes_csv" | xargs)
        NGPU=$(echo "$GPU_number_csv" | xargs)
        # CSV_GPU_BLOCKS=$(echo "$gpu_blocks_csv" | xargs) # Read, but not directly used by main.out call
        SINGLE_TILE_WIDTH=$(echo "$gpu_threads_csv" | xargs)

        # Validate inputs
        if ! [[ "$MSIZE" =~ ^[0-9]+$ ]] || \
           ! [[ "$NTASK" =~ ^[0-9]+$ ]] || \
           ! [[ "$NGPU" =~ ^[0-9]+$ ]] || \
           ! [[ "$SINGLE_TILE_WIDTH" =~ ^[0-9]+$ ]]; then
            echo "  Warning: Invalid data in $csv_filename (row $row_num): '$matrix_width_csv,$processes_csv,$GPU_number_csv,$gpu_blocks_csv,$gpu_threads_csv'. Skipping."
            continue
        fi
        
        if [ "$MSIZE" -le 0 ] || [ "$NTASK" -le 0 ] || [ "$NGPU" -le 0 ] || [ "$SINGLE_TILE_WIDTH" -le 0 ]; then
            echo "  Warning: Non-positive value in $csv_filename (row $row_num): '$matrix_width_csv,$processes_csv,$GPU_number_csv,$gpu_blocks_csv,$gpu_threads_csv'. Skipping."
            continue
        fi


        JOB_NAME_SUFFIX="${csv_filename_no_ext}_N${MSIZE}_T${NTASK}_G${NGPU}_TW${SINGLE_TILE_WIDTH}"
        # If CSVs can have multiple rows, add row_num for full uniqueness:
        # JOB_NAME_SUFFIX="${csv_filename_no_ext}_row${row_num}_N${MSIZE}_T${NTASK}_G${NGPU}_TW${SINGLE_TILE_WIDTH}"
        SLURM_SCRIPT_NAME_TMP="job_${JOB_NAME_SUFFIX}.slurm" # Temporary script name

        echo "  Config from $csv_filename (row $row_num): MSIZE=${MSIZE}, NTASK=${NTASK}, NGPU=${NGPU}, TILE_WIDTH=${SINGLE_TILE_WIDTH}"
        echo "    (Note: gpu_blocks value '$gpu_blocks_csv' from CSV is not directly passed to main.out as it calculates grid dimensions internally)"

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
#SBATCH --job-name=mat_mul_${JOB_NAME_SUFFIX}

srun nsys profile \\
    --force-overwrite true \\
    --gpu-metrics-device=all \\
    --output=profiling/profile_${JOB_NAME_SUFFIX}_procid\\\$SLURM_PROCID \\
    bin/main.out ${MSIZE} 1 ${SINGLE_TILE_WIDTH}
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
done

echo " "
if [ ${#JOBS_IDS[@]} -eq 0 ]; then
    echo "No SLURM jobs were submitted. Check CSV files in tests/ and their content."
    echo "Skipping plot job submission."
else
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
```// filepath: /home/redy1908/HPC-MultiGPU-MatrixMult/scripts/run.sh
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

echo "Starting submission of SLURM jobs based on CSV files in tests/ ..."
CSV_FILES_DIR="tests"

for csv_file_path in "$CSV_FILES_DIR"/*.csv; do
    if [ ! -f "$csv_file_path" ]; then
        echo "Warning: No CSV files found in $CSV_FILES_DIR, or $csv_file_path is not a file."
        continue
    fi

    csv_filename=$(basename "$csv_file_path")
    csv_filename_no_ext="${csv_filename%.*}"
    echo "Processing CSV file: $csv_filename"

    row_num=0
    tail -n +2 "$csv_file_path" | while IFS=, read -r matrix_width processes GPU_number tile_width grid_width grid_height; do
        row_num=$((row_num + 1))

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
done

echo " "
if [ ${#JOBS_IDS[@]} -eq 0 ]; then
    echo "No SLURM jobs were submitted. Check CSV files in tests/ and their content."
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