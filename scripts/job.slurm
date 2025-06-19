#!/bin/bash
#SBATCH -p gpus
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus-per-task=1
#SBATCH --time=00:02:00

#SBATCH hetjob

#SBATCH -p gpus
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus-per-task=2
#SBATCH --time=00:02:00

#SBATCH hetjob

#SBATCH -p gpus
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus-per-task=4
#SBATCH --time=00:02:00

srun --job-name=job0 --output=output0.log --error=error0.log --het-group=0 bin/main.out 512 32 8 8 filename
srun --job-name=job1 --output=output1.log --error=error1.log --het-group=1 bin/main.out 512 32 8 8 filename
srun --job-name=job2 --output=output2.log --error=error2.log --het-group=2 bin/main.out 512 32 8 8 filename
srun --job-name=job3 --output=output3.log --error=error3.log --het-group=2 bin/main.out 1024 32 8 8 filename
