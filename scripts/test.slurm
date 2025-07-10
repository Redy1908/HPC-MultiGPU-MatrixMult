#!/bin/bash
#SBATCH -p gpus
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --output=logs/output.log
#SBATCH --error=logs/error.err
#SBATCH --job-name=test

srun bin/main.out 512 32 1 1 testA2
wait
