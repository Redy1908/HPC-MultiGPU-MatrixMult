#!/bin/bash

#SBATCH -p gpus
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus-per-task=1
#SBATCH --time=00:02:00

#SBATCH hetjob
#SBATCH -p gpus
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus-per-task=1
#SBATCH --time=00:02:00

# TEST a1
srun --output=outputa1.log --open-mode=append --error=errora1.log --het-group=0 bin/main.out 128 32 1 1 filename
srun --output=outputa1.log --open-mode=append --error=errora1.log --het-group=1 bin/main.out 128 32 1 1 filename
# srun --output=outputa1.log --open-mode=append --error=errora1.log --het-group=2 bin/main.out 128 32 1 1 filename

# TEST a2
srun --output=outputa2.log --open-mode=append --error=errora2.log --het-group=1 bin/main.out 512 4 1 1 filename
srun --output=outputa2.log --open-mode=append --error=errora2.log --het-group=1 bin/main.out 512 8 1 1 filename
srun --output=outputa2.log --open-mode=append --error=errora2.log --het-group=1 bin/main.out 512 16 1 1 filename
srun --output=outputa2.log --open-mode=append --error=errora2.log --het-group=1 bin/main.out 512 32 1 1 filename
srun --output=outputa2.log --open-mode=append --error=errora2.log --het-group=1 bin/main.out 512 32 2 2 filename

wait
